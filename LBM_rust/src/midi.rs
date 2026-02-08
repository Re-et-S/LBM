use std::io::{self, Read};
use std::error::Error;
use std::fmt;
use serde::{Serialize, Deserialize};

#[derive(Debug)]
pub enum MidiError {
    Io(io::Error),
    InvalidHeader,
    InvalidTrackHeader,
    InvalidVlq,
    UnknownStatus(u8),
}

impl From<io::Error> for MidiError {
    fn from(err: io::Error) -> MidiError {
        MidiError::Io(err)
    }
}

// Implement Display to satisfy std::error::Error
impl fmt::Display for MidiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for MidiError {}

#[derive(Debug, Clone, PartialEq)]
pub enum MidiEventType {
    NoteOff { note: u8, velocity: u8 },
    NoteOn { note: u8, velocity: u8 },
    PolyAftertouch { note: u8, pressure: u8 },
    ControlChange { controller: u8, value: u8 },
    ProgramChange { program: u8 },
    ChannelAftertouch { pressure: u8 },
    PitchBend { value: u16 },
    // Meta events
    MetaTempo { microseconds_per_beat: u32 },
    MetaTimeSignature { numerator: u8, denominator: u8 },
    MetaEndOfTrack,
    // Catch-all for others we might not strictly need right now
    Unknown { status: u8, data: Vec<u8> },
}

#[derive(Debug, Clone)]
pub struct MidiEvent {
    pub delta_time: u32,
    pub event: MidiEventType,
    pub channel: u8, // 0-15
}

#[derive(Debug)]
pub struct MidiTrack {
    pub events: Vec<MidiEvent>,
}

#[derive(Debug)]
pub struct MidiFile {
    pub format: u16,
    pub num_tracks: u16,
    pub division: u16, // ticks per beat (usually)
    pub tracks: Vec<MidiTrack>,
}

pub fn read_vlq<R: Read>(reader: &mut R) -> Result<u32, MidiError> {
    let mut value: u32 = 0;
    let mut byte_buffer = [0u8; 1];
    
    loop {
        reader.read_exact(&mut byte_buffer)?;
        let byte = byte_buffer[0];
        
        // Check for overflow (max 4 bytes / 28 bits)
        // If the top 7 bits of the current 32-bit accumulator are set, 
        // shifting left by 7 would overflow.
        if (value & 0xFE00_0000) != 0 {
            return Err(MidiError::InvalidVlq);
        }

        value = (value << 7) | (byte & 0x7F) as u32;

        // If the MSB (0x80) is NOT set, this is the last byte
        if (byte & 0x80) == 0 {
            break;
        }
    }

    Ok(value)
}

/// Helper to read a specific number of bytes
pub fn read_bytes<R: Read>(reader: &mut R, count: usize) -> Result<Vec<u8>, MidiError> {
    let mut buf = vec![0u8; count];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

/// Helper to read a big-endian u16
pub fn read_u16<R: Read>(reader: &mut R) -> Result<u16, MidiError> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_be_bytes(buf))
}

/// Helper to read a big-endian u32
pub fn read_u32<R: Read>(reader: &mut R) -> Result<u32, MidiError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

impl MidiFile {
    /// Main entry point: parses a reader into a complete MidiFile struct
    pub fn parse<R: Read>(reader: &mut R) -> Result<MidiFile, MidiError> {
        // 1. Parse MThd Chunk
        let header_chunk_type = read_bytes(reader, 4)?;
        if header_chunk_type != b"MThd" {
            return Err(MidiError::InvalidHeader);
        }

        let header_length = read_u32(reader)?;
        if header_length != 6 {
            // Standard MIDI header is always 6 bytes
            return Err(MidiError::InvalidHeader); 
        }

        let format = read_u16(reader)?;
        let num_tracks = read_u16(reader)?;
        let division = read_u16(reader)?;

        if (division & 0x8000) != 0 {
            // MSB set means SMPTE time code, not supported for this simple parser
            return Err(MidiError::Io(io::Error::new(
                io::ErrorKind::Unsupported,
                "SMPTE time division not supported",
            )));
        }

        let mut tracks = Vec::with_capacity(num_tracks as usize);

        // 2. Parse Tracks
        for _ in 0..num_tracks {
            tracks.push(MidiTrack::parse(reader)?);
        }

        Ok(MidiFile {
            format,
            num_tracks,
            division,
            tracks,
        })
    }
}

impl MidiTrack {
    pub fn parse<R: Read>(reader: &mut R) -> Result<MidiTrack, MidiError> {
        // 1. Check Track Header
        let chunk_type = read_bytes(reader, 4)?;
        if chunk_type != b"MTrk" {
            return Err(MidiError::InvalidTrackHeader);
        }

        let length = read_u32(reader)?;
        
        // We need to limit the reader to 'length' bytes for this track.
        // We use a 'Take' adapter which acts like a specialized view of the stream.
        let mut track_reader = reader.take(length as u64);
        
        let mut events = Vec::new();
        let mut running_status: u8 = 0;

        // We loop until we have consumed exactly 'length' bytes
        // or we hit the End of Track event.
        while track_reader.limit() > 0 {
            let delta_time = read_vlq(&mut track_reader)?;
            
            // Peek at the next byte to check for Running Status
            // Since we can't easily 'ungetc' in a generic reader, handling running status
            // usually requires reading the byte and deciding what it is.
            let mut status_byte_buf = [0u8; 1];
            track_reader.read_exact(&mut status_byte_buf)?;
            let mut status_byte = status_byte_buf[0];

            if status_byte < 0x80 {
                // This is a data byte! Use running status.
                if running_status == 0 {
                    return Err(MidiError::InvalidHeader); // Invalid running status
                }
                status_byte = running_status;
                
                // CRITICAL: We read a data byte but treated it as status.
                // We need to "virtually" put it back or pass it to the handler.
                // Because standard generic Readers don't support `seek(-1)`, 
                // we will pass this "peeked" byte into our event parser helper.
                events.push(parse_event(&mut track_reader, delta_time, status_byte, Some(status_byte_buf[0]))?);
            } else {
                // New status byte
                running_status = if status_byte >= 0xF0 { 0 } else { status_byte };
                events.push(parse_event(&mut track_reader, delta_time, status_byte, None)?);
            }
            
            if let Some(last_event) = events.last() {
                if let MidiEventType::MetaEndOfTrack = last_event.event {
                    break;
                }
            }
        }

        Ok(MidiTrack { events })
    }
}

// Helper to parse the specific event logic
// 'first_data_byte': If we used running status, we already accidentally read the first data byte.
// We pass it in here so the parser can use it instead of reading from the stream.
fn parse_event<R: Read>(
    reader: &mut R, 
    delta_time: u32, 
    status: u8, 
    first_data_byte: Option<u8>
) -> Result<MidiEvent, MidiError> {
        
    // Actually, handling the "peeked" byte is cleaner if we just handle it inside the match branches.
    // Let's define the channel and event type.
    
    let channel = status & 0x0F;
    let event_type = if status >= 0xF0 {
        // System / Meta events
        match status {
            0xFF => {
                // Meta Event
                // If we are here, 'first_data_byte' is None because 0xFF is a status byte.
                // But wait, the logic in 'parse_track' passes 'None' if status >= 0x80.
                // So we read the meta type.
                let mut meta_type_buf = [0u8; 1];
                reader.read_exact(&mut meta_type_buf)?;
                let meta_type = meta_type_buf[0];
                
                let length = read_vlq(reader)?;
                
                // Read the meta payload
                let mut data = vec![0u8; length as usize];
                reader.read_exact(&mut data)?;
                
                match meta_type {
                    0x2F => MidiEventType::MetaEndOfTrack,
                    0x51 => {
                        if data.len() == 3 {
                            let us = ((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32);
                            MidiEventType::MetaTempo { microseconds_per_beat: us }
                        } else {
                            MidiEventType::Unknown { status, data }
                        }
                    },
                    _ => MidiEventType::Unknown { status, data } // Ignore other meta
                }
            },
            _ => {
                // SysEx (0xF0, 0xF7) - Skip them
                let length = read_vlq(reader)?;
                // Discard data
                let mut scratch = vec![0u8; length as usize];
                reader.read_exact(&mut scratch)?;
                MidiEventType::Unknown { status, data: scratch }
            }
        }
    } else {
        // Channel Messages
        let param1 = if let Some(p) = first_data_byte { p } else { 
            let mut b = [0u8; 1];
            reader.read_exact(&mut b)?;
            b[0]
        };

        match status & 0xF0 {
            0x80 => { // Note Off
                let mut p2 = [0u8; 1]; reader.read_exact(&mut p2)?;
                MidiEventType::NoteOff { note: param1, velocity: p2[0] }
            },
            0x90 => { // Note On
                let mut p2 = [0u8; 1]; reader.read_exact(&mut p2)?;
                if p2[0] == 0 {
                    MidiEventType::NoteOff { note: param1, velocity: 0 }
                } else {
                    MidiEventType::NoteOn { note: param1, velocity: p2[0] }
                }
            },
            0xA0 => { // Poly Aftertouch
                let mut p2 = [0u8; 1]; reader.read_exact(&mut p2)?;
                MidiEventType::PolyAftertouch { note: param1, pressure: p2[0] }
            },
            0xB0 => { // Control Change
                let mut p2 = [0u8; 1]; reader.read_exact(&mut p2)?;
                MidiEventType::ControlChange { controller: param1, value: p2[0] }
            },
            0xC0 => { // Program Change
                MidiEventType::ProgramChange { program: param1 }
            },
            0xD0 => { // Channel Aftertouch
                MidiEventType::ChannelAftertouch { pressure: param1 }
            },
            0xE0 => { // Pitch Bend
                let mut p2 = [0u8; 1]; reader.read_exact(&mut p2)?;
                let val = ((p2[0] as u16) << 7) | (param1 as u16);
                MidiEventType::PitchBend { value: val }
            },
            _ => return Err(MidiError::UnknownStatus(status)),
        }
    };

    Ok(MidiEvent {
        delta_time,
        event: event_type,
        channel,
    })
}

#[derive(Debug)]
pub struct UnifiedEvent<'a>{
    pub abs_ticks: u64,
    pub event: &'a MidiEventType,
    pub channel: u8,
}

impl MidiFile {
    pub fn flatten(&self) -> Vec<UnifiedEvent> {
        let mut all_events = Vec::new();

        for track in &self.tracks {
            let mut current_abs_ticks = 0;

            for event in &track.events {
                current_abs_ticks += event.delta_time as u64;

                all_events.push(
                  UnifiedEvent {
                      abs_ticks: current_abs_ticks,
                      event: &event.event,
                      channel: event.channel,
                  }  
                );
            }
        }

        all_events.sort_by_key(|e| e.abs_ticks);

        all_events
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PianoRollChunk {
    pub start_tick: u64,
    pub duration_quarters: f32,
    // Notes where the key is physically held down
    pub keys_active: Vec<u8>,   
    // Notes sounding ONLY because the pedal is holding them (key is up)
    pub sustained_only: Vec<u8>, 
}

impl PianoRollChunk {
    /// Serializes the chunk into a raw 36-byte buffer (Duration + Masks)
    pub fn to_binary(&self) -> [u8; 36] {
        let mut buffer = [0u8; 36];

        // 1. Write Duration as f32 (4 bytes)
        let dur_bytes = self.duration_quarters.to_le_bytes();
        buffer[0..4].copy_from_slice(&dur_bytes);

        // 2. Write Bitmasks (same as before)
        self.pack_pitches(&self.keys_active, &mut buffer[4..20]);
        self.pack_pitches(&self.sustained_only, &mut buffer[20..36]);

        buffer
    }

    // Helper to turn pitch list [60, 64, 67] into bitmask
    fn pack_pitches(&self, pitches: &[u8], out_buf: &mut [u8]) {
        for &pitch in pitches {
            if pitch < 128 {
                let byte_idx = (pitch / 8) as usize;
                let bit_idx = pitch % 8;
                // Set the specific bit
                out_buf[byte_idx] |= 1 << bit_idx;
            }
        }
    }
}

#[derive(Debug)]
pub struct PianoRollState {
    keys_down: [bool; 128],
    pedal_down: bool,
    sustained_buffer: [bool; 128],
}

impl PianoRollState {
    pub fn new() -> Self {
        Self {
            keys_down: [false; 128],
            pedal_down: false,
            sustained_buffer: [false; 128],
        }
    }

    pub fn handle_event(&mut self, event: &MidiEventType) {
        match event {
            MidiEventType::NoteOn {note, velocity} => {
                let n = *note as usize;
                if *velocity > 0 {
                    self.keys_down[n] = true;
                    self.sustained_buffer[n] = false;
                } else {
                    self.handle_note_off(n);
                }
            }
            MidiEventType::NoteOff {note, ..} => {
                self.handle_note_off(*note as usize);
            }
            MidiEventType::ControlChange{controller, value} => {
                if *controller == 64 {
                    // sustain pedal
                    let new_pedal = *value >= 64;

                    if self.pedal_down && !new_pedal{
                        // pedal released. Clear the buffer
                        self.sustained_buffer = [false; 128];
                    }
                    self.pedal_down = new_pedal;
                }               
            }
            _ => {}
        }
    }

    fn handle_note_off(&mut self, n: usize) {
        self.keys_down[n] = false;

        if self.pedal_down {
            self.sustained_buffer[n] = true;
        }
    }

    pub fn capture(&self) -> (Vec<u8>, Vec<u8>) {
        let mut active = Vec::new();
        let mut sustained = Vec::new();

        for i in 0..128 {
            if self.keys_down[i] {
                active.push(i as u8);
            } else if self.sustained_buffer[i] {
                sustained.push(i as u8);
            }
        }
        (active, sustained)
    }
    
}

pub fn events_to_chunks(events: &[UnifiedEvent], ppq: u16) -> Vec<PianoRollChunk> {
    let mut chunks = Vec::new();
    let mut state = PianoRollState::new();
    let mut last_tick = 0;

    let ticks_to_quarters = 1.0 / (ppq as f32);
    
    for event in events {
        let current_tick = event.abs_ticks;
        let delta = current_tick - last_tick;

        if delta > 0 {
            let (keys, sust) = state.capture();

            let duration_q = (delta as f32) * ticks_to_quarters;
            chunks.push(PianoRollChunk {
                start_tick: last_tick,
                duration_quarters: duration_q,
                keys_active: keys,
                sustained_only: sust,
            });

            last_tick = current_tick;
            
        }

        state.handle_event(event.event);
    }

    chunks
}
