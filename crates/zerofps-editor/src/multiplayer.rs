use std::{collections::HashMap, io::ErrorKind, sync::mpsc, thread, time::Duration};

use serde::{Deserialize, Serialize};
use tungstenite::{Message, connect, stream::MaybeTlsStream};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum WireValue {
    Scalar(f32),
    Array(Vec<f32>),
}

impl WireValue {
    fn into_array(self) -> Vec<f32> {
        match self {
            Self::Scalar(value) => vec![value],
            Self::Array(values) => values,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireMessage {
    Join {
        lobby: String,
        player: String,
    },
    Publish {
        lobby: String,
        player: String,
        variable: String,
        value: WireValue,
    },
    State {
        lobby: String,
        values: HashMap<String, HashMap<String, WireValue>>,
    },
    Update {
        lobby: String,
        player: String,
        variable: String,
        value: WireValue,
    },
    PlayerLeft {
        lobby: String,
        player: String,
    },
    Error {
        message: String,
    },
}

pub enum MultiplayerCommand {
    Connect {
        url: String,
        lobby: String,
        player: String,
    },
    Disconnect,
    Publish {
        variable: String,
        values: Vec<f32>,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum MultiplayerEvent {
    Connected,
    Disconnected(String),
    State(HashMap<String, HashMap<String, Vec<f32>>>),
    Update {
        player: String,
        variable: String,
        values: Vec<f32>,
    },
    Error(String),
}

pub struct MultiplayerClient {
    commands: mpsc::Sender<MultiplayerCommand>,
    pub events: mpsc::Receiver<MultiplayerEvent>,
}

impl MultiplayerClient {
    pub fn new() -> Self {
        let (commands, command_rx) = mpsc::channel();
        let (event_tx, events) = mpsc::channel();
        thread::Builder::new()
            .name("zerofps-multiplayer".into())
            .spawn(move || worker(command_rx, event_tx))
            .expect("multiplayer worker should start");
        Self { commands, events }
    }

    pub fn connect(&self, url: String, lobby: String, player: String) {
        let _ = self
            .commands
            .send(MultiplayerCommand::Connect { url, lobby, player });
    }
    pub fn disconnect(&self) {
        let _ = self.commands.send(MultiplayerCommand::Disconnect);
    }
    pub fn publish(&self, variable: String, values: Vec<f32>) {
        let _ = self
            .commands
            .send(MultiplayerCommand::Publish { variable, values });
    }
}

impl Drop for MultiplayerClient {
    fn drop(&mut self) {
        let _ = self.commands.send(MultiplayerCommand::Shutdown);
    }
}

fn worker(commands: mpsc::Receiver<MultiplayerCommand>, events: mpsc::Sender<MultiplayerEvent>) {
    let mut socket = None;
    let mut identity = (String::new(), String::new());
    loop {
        while let Ok(command) = commands.try_recv() {
            match command {
                MultiplayerCommand::Connect { url, lobby, player } => {
                    socket = None;
                    match connect(url.as_str()) {
                        Ok((mut ws, _)) => {
                            if let MaybeTlsStream::Plain(stream) = ws.get_mut() {
                                let _ = stream.set_read_timeout(Some(Duration::from_millis(10)));
                                let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                            }
                            identity = (lobby.clone(), player.clone());
                            let join = WireMessage::Join { lobby, player };
                            if ws
                                .send(Message::Text(serde_json::to_string(&join).unwrap().into()))
                                .is_ok()
                            {
                                socket = Some(ws);
                                let _ = events.send(MultiplayerEvent::Connected);
                            }
                        }
                        Err(error) => {
                            let _ = events.send(MultiplayerEvent::Error(error.to_string()));
                        }
                    }
                }
                MultiplayerCommand::Disconnect => {
                    if let Some(mut ws) = socket.take() {
                        let _ = ws.close(None);
                    }
                    let _ = events.send(MultiplayerEvent::Disconnected("Disconnected".into()));
                }
                MultiplayerCommand::Publish { variable, values } => {
                    if let Some(ws) = socket.as_mut() {
                        let message = WireMessage::Publish {
                            lobby: identity.0.clone(),
                            player: identity.1.clone(),
                            variable,
                            value: WireValue::Array(values),
                        };
                        if let Ok(json) = serde_json::to_string(&message) {
                            let _ = ws.send(Message::Text(json.into()));
                        }
                    }
                }
                MultiplayerCommand::Shutdown => return,
            }
        }
        if let Some(ws) = socket.as_mut() {
            match ws.read() {
                Ok(Message::Text(text)) => match serde_json::from_str::<WireMessage>(&text) {
                    Ok(WireMessage::State { values, .. }) => {
                        let values = values
                            .into_iter()
                            .map(|(player, variables)| {
                                (
                                    player,
                                    variables
                                        .into_iter()
                                        .map(|(name, value)| (name, value.into_array()))
                                        .collect(),
                                )
                            })
                            .collect();
                        let _ = events.send(MultiplayerEvent::State(values));
                    }
                    Ok(WireMessage::Update {
                        player,
                        variable,
                        value,
                        ..
                    }) => {
                        let _ = events.send(MultiplayerEvent::Update {
                            player,
                            variable,
                            values: value.into_array(),
                        });
                    }
                    Ok(WireMessage::Error { message }) => {
                        let _ = events.send(MultiplayerEvent::Error(message));
                    }
                    _ => {}
                },
                Ok(Message::Close(_)) => {
                    socket = None;
                    let _ = events.send(MultiplayerEvent::Disconnected(
                        "Server closed connection".into(),
                    ));
                }
                Err(tungstenite::Error::Io(error))
                    if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                Err(error) => {
                    socket = None;
                    let _ = events.send(MultiplayerEvent::Disconnected(error.to_string()));
                }
                _ => {}
            }
        } else {
            thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publish_wire_message_round_trips() {
        let message = WireMessage::Publish {
            lobby: "test".into(),
            player: "p1".into(),
            variable: "speed".into(),
            value: WireValue::Array(vec![12.5, -3.0, 8.0]),
        };
        let json = serde_json::to_string(&message).unwrap();
        assert_eq!(serde_json::from_str::<WireMessage>(&json).unwrap(), message);
    }

    #[test]
    fn legacy_scalar_update_deserializes_as_one_element_array() {
        let message: WireMessage = serde_json::from_str(
            r#"{"type":"update","lobby":"test","player":"p1","variable":"x","value":2.5}"#,
        )
        .unwrap();
        let WireMessage::Update { value, .. } = message else {
            panic!("expected update")
        };
        assert_eq!(value.into_array(), vec![2.5]);
    }
}
