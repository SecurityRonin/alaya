use rusqlite::{params, Connection};
use crate::error::Result;
use crate::types::*;

pub fn store_impression(conn: &Connection, imp: &NewImpression) -> Result<ImpressionId> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    conn.execute(
        "INSERT INTO impressions (domain, observation, valence, timestamp)
         VALUES (?1, ?2, ?3, ?4)",
        params![imp.domain, imp.observation, imp.valence, now],
    )?;
    Ok(ImpressionId(conn.last_insert_rowid()))
}

pub fn get_impressions_by_domain(conn: &Connection, domain: &str, limit: u32) -> Result<Vec<Impression>> {
    let mut stmt = conn.prepare(
        "SELECT id, domain, observation, valence, timestamp
         FROM impressions WHERE domain = ?1
         ORDER BY timestamp DESC LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![domain, limit], |row| {
        Ok(Impression {
            id: ImpressionId(row.get(0)?),
            domain: row.get(1)?,
            observation: row.get(2)?,
            valence: row.get(3)?,
            timestamp: row.get(4)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn count_impressions_by_domain(conn: &Connection, domain: &str) -> Result<u64> {
    let count: i64 = conn.query_row(
        "SELECT count(*) FROM impressions WHERE domain = ?1",
        [domain],
        |row| row.get(0),
    )?;
    Ok(count as u64)
}

pub fn store_preference(conn: &Connection, domain: &str, preference: &str, confidence: f32) -> Result<PreferenceId> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    conn.execute(
        "INSERT INTO preferences (domain, preference, confidence, evidence_count, first_observed, last_reinforced)
         VALUES (?1, ?2, ?3, 1, ?4, ?4)",
        params![domain, preference, confidence, now],
    )?;
    Ok(PreferenceId(conn.last_insert_rowid()))
}

pub fn get_preferences(conn: &Connection, domain: Option<&str>) -> Result<Vec<Preference>> {
    let (sql, param): (&str, Option<&str>) = match domain {
        Some(d) => (
            "SELECT id, domain, preference, confidence, evidence_count, first_observed, last_reinforced
             FROM preferences WHERE domain = ?1 ORDER BY confidence DESC",
            Some(d),
        ),
        None => (
            "SELECT id, domain, preference, confidence, evidence_count, first_observed, last_reinforced
             FROM preferences ORDER BY confidence DESC",
            None,
        ),
    };
    let mut stmt = conn.prepare(sql)?;
    let rows = if let Some(d) = param {
        stmt.query_map([d], map_preference)?
    } else {
        stmt.query_map([], map_preference)?
    };
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn map_preference(row: &rusqlite::Row<'_>) -> rusqlite::Result<Preference> {
    Ok(Preference {
        id: PreferenceId(row.get(0)?),
        domain: row.get(1)?,
        preference: row.get(2)?,
        confidence: row.get(3)?,
        evidence_count: row.get(4)?,
        first_observed: row.get(5)?,
        last_reinforced: row.get(6)?,
    })
}

pub fn reinforce_preference(conn: &Connection, id: PreferenceId, additional_evidence: u32) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    conn.execute(
        "UPDATE preferences SET evidence_count = evidence_count + ?2,
                last_reinforced = ?3,
                confidence = MIN(1.0, confidence + 0.1 * ?2)
         WHERE id = ?1",
        params![id.0, additional_evidence, now],
    )?;
    Ok(())
}

pub fn decay_preferences(conn: &Connection, now: i64, half_life_secs: i64) -> Result<u64> {
    // Exponential decay: confidence *= exp(-0.693 * age / half_life)
    // SQLite doesn't have exp(), so we approximate with a linear decay per sweep
    // Actually, we can compute the factor and multiply:
    let changed = conn.execute(
        "UPDATE preferences SET confidence = confidence * 0.95
         WHERE (?1 - last_reinforced) > ?2 AND confidence > 0.01",
        params![now, half_life_secs],
    )?;
    Ok(changed as u64)
}

pub fn prune_weak_preferences(conn: &Connection, min_confidence: f32) -> Result<u64> {
    let deleted = conn.execute(
        "DELETE FROM preferences WHERE confidence < ?1",
        [min_confidence],
    )?;
    Ok(deleted as u64)
}

pub fn prune_old_impressions(conn: &Connection, max_age_secs: i64) -> Result<u64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let cutoff = now - max_age_secs;
    let deleted = conn.execute(
        "DELETE FROM impressions WHERE timestamp < ?1",
        [cutoff],
    )?;
    Ok(deleted as u64)
}

pub fn count_preferences(conn: &Connection) -> Result<u64> {
    let count: i64 = conn.query_row("SELECT count(*) FROM preferences", [], |row| row.get(0))?;
    Ok(count as u64)
}

pub fn count_impressions(conn: &Connection) -> Result<u64> {
    let count: i64 = conn.query_row("SELECT count(*) FROM impressions", [], |row| row.get(0))?;
    Ok(count as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    #[test]
    fn test_impressions_crud() {
        let conn = open_memory_db().unwrap();
        let id = store_impression(&conn, &NewImpression {
            domain: "communication".to_string(),
            observation: "user prefers short answers".to_string(),
            valence: 1.0,
        }).unwrap();
        assert_eq!(id.0, 1);
        let imps = get_impressions_by_domain(&conn, "communication", 10).unwrap();
        assert_eq!(imps.len(), 1);
        assert_eq!(imps[0].observation, "user prefers short answers");
    }

    #[test]
    fn test_preferences_crud() {
        let conn = open_memory_db().unwrap();
        let id = store_preference(&conn, "style", "concise answers", 0.7).unwrap();
        let prefs = get_preferences(&conn, Some("style")).unwrap();
        assert_eq!(prefs.len(), 1);
        assert_eq!(prefs[0].preference, "concise answers");

        reinforce_preference(&conn, id, 2).unwrap();
        let prefs = get_preferences(&conn, Some("style")).unwrap();
        assert_eq!(prefs[0].evidence_count, 3);
        assert!(prefs[0].confidence > 0.7);
    }
}
