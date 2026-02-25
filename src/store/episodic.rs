use rusqlite::{params, Connection};
use crate::error::{AlayaError, Result};
use crate::types::*;

pub fn store_episode(conn: &Connection, ep: &NewEpisode) -> Result<EpisodeId> {
    let ctx_json = serde_json::to_string(&ep.context)?;
    conn.execute(
        "INSERT INTO episodes (content, role, session_id, timestamp, context_json)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![ep.content, ep.role.as_str(), ep.session_id, ep.timestamp, ctx_json],
    )?;
    Ok(EpisodeId(conn.last_insert_rowid()))
}

pub fn get_episode(conn: &Connection, id: EpisodeId) -> Result<Episode> {
    conn.query_row(
        "SELECT id, content, role, session_id, timestamp, context_json
         FROM episodes WHERE id = ?1",
        [id.0],
        |row| {
            let ctx_str: String = row.get(5)?;
            Ok(Episode {
                id: EpisodeId(row.get(0)?),
                content: row.get(1)?,
                role: Role::from_str(&row.get::<_, String>(2)?).unwrap_or(Role::User),
                session_id: row.get(3)?,
                timestamp: row.get(4)?,
                context: serde_json::from_str(&ctx_str).unwrap_or_default(),
            })
        },
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => {
            AlayaError::NotFound(format!("episode {}", id.0))
        }
        other => AlayaError::Db(other),
    })
}

pub fn get_episodes_by_session(conn: &Connection, session_id: &str) -> Result<Vec<Episode>> {
    let mut stmt = conn.prepare(
        "SELECT id, content, role, session_id, timestamp, context_json
         FROM episodes WHERE session_id = ?1 ORDER BY timestamp ASC",
    )?;
    let rows = stmt.query_map([session_id], |row| {
        let ctx_str: String = row.get(5)?;
        Ok(Episode {
            id: EpisodeId(row.get(0)?),
            content: row.get(1)?,
            role: Role::from_str(&row.get::<_, String>(2)?).unwrap_or(Role::User),
            session_id: row.get(3)?,
            timestamp: row.get(4)?,
            context: serde_json::from_str(&ctx_str).unwrap_or_default(),
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn get_recent_episodes(conn: &Connection, limit: u32) -> Result<Vec<Episode>> {
    let mut stmt = conn.prepare(
        "SELECT id, content, role, session_id, timestamp, context_json
         FROM episodes ORDER BY timestamp DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit], |row| {
        let ctx_str: String = row.get(5)?;
        Ok(Episode {
            id: EpisodeId(row.get(0)?),
            content: row.get(1)?,
            role: Role::from_str(&row.get::<_, String>(2)?).unwrap_or(Role::User),
            session_id: row.get(3)?,
            timestamp: row.get(4)?,
            context: serde_json::from_str(&ctx_str).unwrap_or_default(),
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn get_unconsolidated_episodes(conn: &Connection, limit: u32) -> Result<Vec<Episode>> {
    // Episodes not referenced by any semantic node's source_episodes_json
    // Simple approach: episodes whose id is not in any semantic_node source list
    // For now, use a flag approach: episodes not linked to any semantic node via the graph
    let mut stmt = conn.prepare(
        "SELECT e.id, e.content, e.role, e.session_id, e.timestamp, e.context_json
         FROM episodes e
         WHERE NOT EXISTS (
             SELECT 1 FROM links l
             WHERE l.target_type = 'episode' AND l.target_id = e.id
               AND l.source_type = 'semantic'
         )
         AND NOT EXISTS (
             SELECT 1 FROM links l
             WHERE l.source_type = 'episode' AND l.source_id = e.id
               AND l.target_type = 'semantic'
         )
         ORDER BY e.timestamp ASC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit], |row| {
        let ctx_str: String = row.get(5)?;
        Ok(Episode {
            id: EpisodeId(row.get(0)?),
            content: row.get(1)?,
            role: Role::from_str(&row.get::<_, String>(2)?).unwrap_or(Role::User),
            session_id: row.get(3)?,
            timestamp: row.get(4)?,
            context: serde_json::from_str(&ctx_str).unwrap_or_default(),
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn delete_episodes(conn: &Connection, ids: &[EpisodeId]) -> Result<u64> {
    if ids.is_empty() {
        return Ok(0);
    }
    let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
    let sql = format!(
        "DELETE FROM episodes WHERE id IN ({})",
        placeholders.join(",")
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> =
        ids.iter().map(|id| &id.0 as &dyn rusqlite::types::ToSql).collect();
    let count = stmt.execute(params.as_slice())?;
    Ok(count as u64)
}

pub fn count_episodes(conn: &Connection) -> Result<u64> {
    let count: i64 = conn.query_row("SELECT count(*) FROM episodes", [], |row| row.get(0))?;
    Ok(count as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::open_memory_db;

    fn make_episode(content: &str, ts: i64) -> NewEpisode {
        NewEpisode {
            content: content.to_string(),
            role: Role::User,
            session_id: "test-session".to_string(),
            timestamp: ts,
            context: EpisodeContext::default(),
            embedding: None,
        }
    }

    #[test]
    fn test_store_and_get() {
        let conn = open_memory_db().unwrap();
        let id = store_episode(&conn, &make_episode("hello world", 1000)).unwrap();
        let ep = get_episode(&conn, id).unwrap();
        assert_eq!(ep.content, "hello world");
        assert_eq!(ep.role, Role::User);
    }

    #[test]
    fn test_get_by_session() {
        let conn = open_memory_db().unwrap();
        store_episode(&conn, &make_episode("msg1", 1000)).unwrap();
        store_episode(&conn, &make_episode("msg2", 2000)).unwrap();
        let eps = get_episodes_by_session(&conn, "test-session").unwrap();
        assert_eq!(eps.len(), 2);
        assert_eq!(eps[0].content, "msg1");
    }

    #[test]
    fn test_count_and_delete() {
        let conn = open_memory_db().unwrap();
        let id1 = store_episode(&conn, &make_episode("a", 1000)).unwrap();
        let id2 = store_episode(&conn, &make_episode("b", 2000)).unwrap();
        assert_eq!(count_episodes(&conn).unwrap(), 2);
        delete_episodes(&conn, &[id1, id2]).unwrap();
        assert_eq!(count_episodes(&conn).unwrap(), 0);
    }
}
