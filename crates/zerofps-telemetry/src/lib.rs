//! Structured, bounded communication between the editor and a play session.
//!
//! This prototype deliberately uses in-process channels.  The protocol types are
//! serializable and transport-independent, so a loopback socket can replace the
//! channel without changing editor or player messages.

use serde::{Deserialize, Serialize};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel},
    },
    time::{SystemTime, UNIX_EPOCH},
};

pub const PROTOCOL_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Envelope<T> {
    pub protocol_version: u16,
    pub session_id: String,
    pub sequence: u64,
    pub timestamp_millis: u64,
    pub payload: T,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum Event {
    Hello {
        project: String,
        build_id: String,
        process_id: u32,
    },
    Log {
        level: LogLevel,
        target: String,
        message: String,
    },
    FrameStatistics {
        frame: u64,
        frame_time_ms: f32,
        draw_calls: u32,
        triangles: u64,
    },
    State {
        state: PlayState,
    },
    Panic {
        message: String,
        location: Option<String>,
    },
    Exited {
        code: Option<i32>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlayState {
    Starting,
    Running,
    Paused,
    Stopping,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum Command {
    Pause,
    Resume,
    Step { ticks: u32 },
    Stop,
    SetLogLevel { minimum: LogLevel },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SendOutcome {
    Sent,
    DroppedFull,
    Disconnected,
}

#[derive(Debug, Default)]
struct Counters {
    event_sequence: AtomicU64,
    command_sequence: AtomicU64,
    dropped_events: AtomicU64,
    dropped_commands: AtomicU64,
}

pub struct EditorEndpoint {
    session_id: Arc<str>,
    command_tx: SyncSender<Envelope<Command>>,
    event_rx: Receiver<Envelope<Event>>,
    counters: Arc<Counters>,
}

pub struct GameEndpoint {
    session_id: Arc<str>,
    event_tx: SyncSender<Envelope<Event>>,
    command_rx: Receiver<Envelope<Command>>,
    counters: Arc<Counters>,
}

pub fn local_session(
    session_id: impl Into<String>,
    capacity: usize,
) -> (EditorEndpoint, GameEndpoint) {
    assert!(capacity > 0, "telemetry channel capacity must be non-zero");
    let session_id: Arc<str> = Arc::from(session_id.into());
    let counters = Arc::new(Counters::default());
    let (event_tx, event_rx) = sync_channel(capacity);
    let (command_tx, command_rx) = sync_channel(capacity);
    (
        EditorEndpoint {
            session_id: session_id.clone(),
            command_tx,
            event_rx,
            counters: counters.clone(),
        },
        GameEndpoint {
            session_id,
            event_tx,
            command_rx,
            counters,
        },
    )
}

impl EditorEndpoint {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn send(&self, command: Command) -> SendOutcome {
        let sequence = self
            .counters
            .command_sequence
            .fetch_add(1, Ordering::Relaxed);
        send_bounded(
            &self.command_tx,
            envelope(&self.session_id, sequence, command),
            &self.counters.dropped_commands,
        )
    }

    pub fn try_recv(&self) -> Result<Envelope<Event>, TryRecvError> {
        self.event_rx.try_recv()
    }

    pub fn dropped_events(&self) -> u64 {
        self.counters.dropped_events.load(Ordering::Relaxed)
    }
}

impl GameEndpoint {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn send(&self, event: Event) -> SendOutcome {
        let sequence = self.counters.event_sequence.fetch_add(1, Ordering::Relaxed);
        send_bounded(
            &self.event_tx,
            envelope(&self.session_id, sequence, event),
            &self.counters.dropped_events,
        )
    }

    pub fn try_recv(&self) -> Result<Envelope<Command>, TryRecvError> {
        self.command_rx.try_recv()
    }

    pub fn dropped_commands(&self) -> u64 {
        self.counters.dropped_commands.load(Ordering::Relaxed)
    }
}

fn envelope<T>(session_id: &str, sequence: u64, payload: T) -> Envelope<T> {
    Envelope {
        protocol_version: PROTOCOL_VERSION,
        session_id: session_id.to_owned(),
        sequence,
        timestamp_millis: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX),
        payload,
    }
}

fn send_bounded<T>(sender: &SyncSender<T>, value: T, dropped: &AtomicU64) -> SendOutcome {
    match sender.try_send(value) {
        Ok(()) => SendOutcome::Sent,
        Err(TrySendError::Full(_)) => {
            dropped.fetch_add(1, Ordering::Relaxed);
            SendOutcome::DroppedFull
        }
        Err(TrySendError::Disconnected(_)) => SendOutcome::Disconnected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_and_commands_cross_in_order() {
        let (editor, game) = local_session("play-1", 4);
        assert_eq!(
            game.send(Event::State {
                state: PlayState::Running
            }),
            SendOutcome::Sent
        );
        assert_eq!(
            game.send(Event::State {
                state: PlayState::Paused
            }),
            SendOutcome::Sent
        );
        assert_eq!(editor.try_recv().unwrap().sequence, 0);
        assert_eq!(editor.try_recv().unwrap().sequence, 1);

        assert_eq!(editor.send(Command::Step { ticks: 1 }), SendOutcome::Sent);
        let message = game.try_recv().unwrap();
        assert_eq!(message.session_id, "play-1");
        assert_eq!(message.payload, Command::Step { ticks: 1 });
    }

    #[test]
    fn full_channel_drops_instead_of_blocking() {
        let (editor, game) = local_session("bounded", 1);
        assert_eq!(
            game.send(Event::State {
                state: PlayState::Running
            }),
            SendOutcome::Sent
        );
        assert_eq!(
            game.send(Event::State {
                state: PlayState::Paused
            }),
            SendOutcome::DroppedFull
        );
        assert_eq!(editor.dropped_events(), 1);
    }

    #[test]
    fn protocol_is_json_serializable_and_versioned() {
        let (_, game) = local_session("json", 1);
        game.send(Event::Log {
            level: LogLevel::Info,
            target: "game".into(),
            message: "ready".into(),
        });
        // Verify payload itself; envelopes additionally carry the protocol number.
        let json = serde_json::to_string(&Command::Pause).unwrap();
        assert!(json.contains("pause"));
        assert_eq!(PROTOCOL_VERSION, 1);
    }
}
