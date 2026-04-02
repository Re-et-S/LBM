use std::fs::File;
use std::io::{self, Read, BufReader, BufWriter, Write};
use std::path::Path;
use rayon::prelude::*;
use rustc_hash::{FxHashSet, FxHashMap};
use std::collections::BinaryHeap;
use rand::Rng;

const CHUNK_SIZE: usize = 36;
const DURATION_OFFSET: usize = 0;

const MUSICAL_GRID: &[f32] = &[
    8.0,
    4.0,        // Whole Note
    3.0,        // Dotted Half
    2.0,        // Half Note
    2.0/3.0,    // Half Note Triplet
    1.5,        // Dotted Quarter
    1.0,        // Quarter Note
    0.75,       // Dotted Eighth
    0.66666,    // Quarter Triplet (2/3)
    0.5,        // Eighth Note
    0.375,      // Dotted Sixteenth
    0.33333,    // Eighth Triplet
    0.25,       // Sixteenth Note
    0.16666,    // Sixteenth Triplet
    0.125,      // Thirty-Second Note
    0.0625,     // Sixty-fourth Note
    0.03125    
];

pub fn load_all_data<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();

    // Read everything into memory
    reader.read_to_end(&mut buffer)?;

    // Basic validation
    if buffer.len() % CHUNK_SIZE != 0 {
        eprintln!("Warning: File size is not a multiple of {}. Data might be truncated.", CHUNK_SIZE);
    }

    Ok(buffer)
}

pub fn quantize_centroids(centroids: &mut Vec<f32>) { 
    println!("Snapping centroids to musical grid...");

    for centroid in centroids.iter_mut() {
        let original = *centroid;
        let mut best_match = MUSICAL_GRID[0];
        let mut min_diff = f32::MAX;

        for &grid_val in MUSICAL_GRID {
            let diff = (original - grid_val).abs();
            if diff < min_diff {
                min_diff = diff;
                best_match = grid_val;
            }
        }
        *centroid = best_match;
    }

    // We use partial_cmp because f32 doesn't implement Ord (due to NaN)
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // de-duplicate
    centroids.dedup();

    println!("Reduced to {} unique musical durations.", centroids.len());
}

pub fn run_clustering_pipeline(binary_path: &str) -> io::Result<Vec<u8>> {
    println!("Loading raw binary from {}...", binary_path);
    
    let mut raw_data = load_all_data(binary_path)?;

    println!("Extracting durations for analysis...");
    let durations: Vec<f32> = raw_data
        .chunks_exact(CHUNK_SIZE)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk[DURATION_OFFSET..DURATION_OFFSET + 4]
                .try_into()
                .expect("Chunk size is correct");
            f32::from_le_bytes(bytes)
        })
        .collect();

    const N_CLUSTERS: usize = 16;
    const MAX_ITERATIONS: usize = 100;
    const TOLERANCE: f32 = 1e-4;

    let start: f32 = 1.0 / 64.0;
    let end: f32 = 4.0;
    let base_ratio = end / start;

    let mut centroids: [f32; N_CLUSTERS] = std::array::from_fn(|i| {
        let progress = (i as f32) / ((N_CLUSTERS - 1) as f32);
        start * base_ratio.powf(progress)
    });

    println!("Initial centroids: {:.4?}", centroids);

    for iteration in 0..MAX_ITERATIONS {
        
        let assignments: Vec<usize> = durations.par_iter()
            .map(|&duration| {
                let mut best_index = 0;
                let mut min_distance = f32::MAX;

                for (i, &centroid) in centroids.iter().enumerate() {
                    let distance = (duration - centroid).abs();
                    if distance < min_distance {
                        min_distance = distance;
                        best_index = i;
                    }
                }
                best_index
            })
            .collect();

        let mut new_centroids = [0.0f32; N_CLUSTERS];
        let mut counts = [0usize; N_CLUSTERS];

        for (&duration, &cluster_idx) in durations.iter().zip(&assignments) {
            new_centroids[cluster_idx] += duration;
            counts[cluster_idx] += 1;
        }

        for i in 0..N_CLUSTERS {
            if counts[i] > 0 {
                new_centroids[i] /= counts[i] as f32;
            } else {
                // Orphan cluster: Keep old position
                new_centroids[i] = centroids[i];
            }
        }

        // Convergence
        let mut shift: f32 = 0.0;
        for i in 0..N_CLUSTERS {
            shift += (centroids[i] - new_centroids[i]).abs();
        }

        println!("Iteration {}: Shift = {:.6}", iteration, shift);

        if shift < TOLERANCE {
            println!("Converged at iteration {}!", iteration);
            centroids = new_centroids;
            break;
        }

        centroids = new_centroids;
    }

    // snap to grid
    println!("K-means clustering centroids: {:.4?}", centroids);
    println!("Quantizing binary data...");

    let mut centroid_vec = centroids.to_vec();
    quantize_centroids(&mut centroid_vec);
    
    raw_data.par_chunks_exact_mut(CHUNK_SIZE)
        .for_each(|chunk| {
            let current_bytes: [u8; 4] = chunk[DURATION_OFFSET..DURATION_OFFSET + 4]
                .try_into()
                .unwrap();
            let current_duration = f32::from_le_bytes(current_bytes);

            let mut best_centroid = centroids[0];
            let mut min_dist = f32::MAX;

            for &centroid in &centroid_vec {
                let dist = (current_duration - centroid).abs();
                if dist < min_dist {
                    min_dist = dist;
                    best_centroid = centroid;
                }
            }

            let new_bytes = best_centroid.to_le_bytes();
            chunk[DURATION_OFFSET..DURATION_OFFSET + 4].copy_from_slice(&new_bytes);
        });

    println!("Quantization complete.");
    println!("Final Centroids: {:.4?}", centroid_vec);
    
    // Return the modified binary ready for BPE
    Ok(raw_data)
}

pub fn tokenize_data(raw_data: &[u8]) -> (Vec<u32>, Vec<[u8; CHUNK_SIZE]>) {
    let unique_chunks: FxHashSet<[u8; CHUNK_SIZE]> = raw_data
        .par_chunks_exact(CHUNK_SIZE)
        .fold(
            || FxHashSet::default(), // Init: Create the empty local set
            |mut set, chunk| {       // Action: Add items to that SAME local set
                let fixed_chunk: [u8; CHUNK_SIZE] = chunk.try_into().unwrap();
                set.insert(fixed_chunk);
                set // Pass the accumulated set to the next iteration
            }
        )
        .reduce(
            || FxHashSet::default(),
            |mut a, b| {
                a.extend(b);
                a
            },
        );
    println!("Found {} unique tokens.", unique_chunks.len());

    let mut vocab: Vec<[u8; CHUNK_SIZE]> = unique_chunks.into_iter().collect();
    vocab.par_sort_unstable();

    let token_to_id: FxHashMap<[u8; CHUNK_SIZE], u32> = vocab
        .iter()
        .enumerate()
        .map(|(id, &chunk)| (chunk, id as u32))
        .collect();

    let tokenized_data: Vec<u32> = raw_data
        .par_chunks_exact(CHUNK_SIZE)
        .map(|chunk|{
            let fixed_chunk: [u8; CHUNK_SIZE] = chunk.try_into().unwrap();
            *token_to_id.get(&fixed_chunk).expect("Chunk must exist in vocab")
        })
        .collect();

    (tokenized_data, vocab)
}

const NULL_INDEX: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)] // Copy is cheap (12 bytes)
struct Node {
    prev: u32,
    next: u32,
    val: u32, // The Token ID
}

pub fn build_linked_list(tokens: &[u32]) -> Vec<Node> {
    let len = tokens.len();
    let mut nodes = Vec::with_capacity(len);

    for (i, &token) in tokens.iter().enumerate() {
        let prev = if i==0 {NULL_INDEX} else { (i-1) as u32 };
        let next = if i==len-1 {NULL_INDEX} else {(i+1) as u32};

        nodes.push(Node {
           prev,
           next,
           val:token,
        });
    }
    nodes
}

pub fn run_bpe(tokenized_data: Vec<u32>, vocab_size: usize) -> (Vec<u32>, Vec<(u32, u32, u32)>) {
    let mut nodes = build_linked_list(&tokenized_data);
    let mut pair_counts = FxHashMap::default();
    let mut locations: Vec<Vec<u32>> = vec![Vec::new(); vocab_size+vocab_size]; // pre_allocate
    let mut merge_rules: Vec<(u32, u32, u32)> = Vec::new();
    
    // locations list to track the position of a token in the linked list
    for (i, node) in nodes.iter().enumerate() {
        let val = node.val as usize;
        
        // A. Update Locations
        if val >= locations.len() {
            locations.resize(val + 1000, Vec::new()); // Resize with buffer
        }
        locations[val].push(i as u32);

        // B. Update Pair Counts (only if not end of list)
        if node.next != NULL_INDEX {
            let next_val = nodes[node.next as usize].val;
            *pair_counts.entry((node.val, next_val)).or_insert(0) += 1;
        }
    }

    println!("Found {} unique pairs.", pair_counts.len());

    // priority queue
    let mut heap = BinaryHeap::with_capacity(pair_counts.len());

    for (&(left,right), &count) in &pair_counts {
        heap.push((count, left, right));
    }

    let mut merges_completed = 0;
    loop {
        let (heap_count, left_token, right_token) = loop {
            match heap.pop() {
                None => break (0,0,0),
                Some((count, left, right)) => {
                    match pair_counts.get(&(left,right)) {
                        Some(&real_count) if real_count == count => {
                            break (count, left, right);
                        }
                        _ => continue,
                    }
                }
            }    
        };

        if heap_count < 2 {
            println!("Stopping: No pairs occur more than once.");
            break;
        }

        let new_token_id = (vocab_size + merges_completed) as u32;
        merges_completed += 1;
        merge_rules.push((new_token_id, left_token, right_token));

        if merges_completed % 1000 == 0 {
            println!("Merge {}: {} + {} -> {} (Count: {})", 
                 merges_completed, left_token, right_token, new_token_id, heap_count);
        }

        if new_token_id as usize >= locations.len() {
            locations.resize(new_token_id as usize + 1000, Vec::new());
        }

        let occurrences = locations[left_token as usize].clone();

        for &index in &occurrences {
            // need to check if the pattern still exists
            let current_node = nodes[index as usize];
            if current_node.val != left_token || current_node.next == NULL_INDEX{
                continue;
            }

            let next_idx = current_node.next as usize;
            let next_node = nodes[next_idx];

            if next_node.val != right_token {
                continue;
            }

            // identify neighbors for count updates
            let prev_idx = current_node.prev;
            let far_right_idx = next_node.next;

            if prev_idx != NULL_INDEX {
                let prev_val = nodes[prev_idx as usize].val;
                if let Some(c) = pair_counts.get_mut(&(prev_val, left_token)) {
                    *c -= 1;
                }
            }

            if far_right_idx != NULL_INDEX {
                let far_right_val = nodes[far_right_idx as usize].val;
                if let Some(c) = pair_counts.get_mut(&(right_token, far_right_val)) {
                    *c -= 1;
                }
            }
            
            // 1. Update the Left Node to become the New Token
            nodes[index as usize].val = new_token_id;
            nodes[index as usize].next = far_right_idx;
            
            // 2. Link the Far Right node back to the New Token
            if far_right_idx != NULL_INDEX {
                nodes[far_right_idx as usize].prev = index as u32;
            }
            
            // make the tombstone for debugging
            nodes[next_idx].val = u32::MAX;
            nodes[next_idx].prev = NULL_INDEX;
            nodes[next_idx].next = NULL_INDEX;

            if prev_idx != NULL_INDEX {
                let prev_val = nodes[prev_idx as usize].val;
                if prev_val != u32::MAX {
                    let new_pair = (prev_val, new_token_id);
                    let count = pair_counts.entry(new_pair).or_insert(0);
                    *count += 1;
                    heap.push((*count, prev_val, new_token_id));
                }
            }

            if far_right_idx != NULL_INDEX {
                let far_right_val = nodes[far_right_idx as usize].val;
                if far_right_val != u32::MAX {
                    let new_pair = (new_token_id, far_right_val);
                    let count = pair_counts.entry(new_pair).or_insert(0);
                    *count += 1;
                    heap.push((*count, new_token_id, far_right_val));
                }
            }

            locations[new_token_id as usize].push(index);
            
        }

        pair_counts.remove(&(left_token, right_token));

        
    }
    
    println!("BPE completed.");

    let mut head_idx = NULL_INDEX;
    for (i, node) in nodes.iter().enumerate() {
        // Find a node that has no previous neighbor AND is not a deleted tombstone
        if node.prev == NULL_INDEX && node.val != u32::MAX {
            head_idx = i as u32;
            break;
        }
    }

    if head_idx == NULL_INDEX {
        panic!("Critical Error: Linked List is empty or circular!");
    }

    let mut compressed_data = Vec::with_capacity(tokenized_data.len());
    let mut curr = head_idx;

    while curr != NULL_INDEX {
        let node = nodes[curr as usize];
        compressed_data.push(node.val);
        curr = node.next;
    }
    println!("Original Length: {}, Compressed Length: {}", 
        tokenized_data.len(), compressed_data.len());
    
    (compressed_data, merge_rules)
}

// Converts the linear list of rules into a fast lookup map
pub fn build_decoding_map(merge_rules: &[(u32, u32, u32)]) -> FxHashMap<u32, (u32, u32)> {
    merge_rules.iter()
        .map(|&(id, left, right)| (id, (left, right)))
        .collect()
}

fn decode_recursive(
    token: u32, 
    vocab: &Vec<[u8; 36]>,
    rule_map: &FxHashMap<u32, (u32, u32)>, 
    output: &mut Vec<[u8; 36]> // Append results here
) {

    if (token as usize) < vocab.len() {
        output.push(vocab[token as usize]);
    } else {
        match rule_map.get(&token) {
            Some(&(left, right)) => {
                decode_recursive(left, vocab, rule_map, output);
                decode_recursive(right, vocab, rule_map, output);
            },
            None => {
                eprintln!("Warning: Token {} not found in rules or vocab! Skipping.", token);
            }
        }
    }
}

pub fn decode_sequence(
    sequence: &[u32], 
    vocab: &Vec<[u8; 36]>, 
    merge_rules: &Vec<(u32, u32, u32)>
) -> Vec<[u8; 36]> {
    let rule_map = build_decoding_map(merge_rules);
    let mut decoded_chunks = Vec::new();

    for &token in sequence {
        decode_recursive(token, vocab, &rule_map, &mut decoded_chunks);
    }
    
    decoded_chunks
}

pub fn generate_sequence(
    compressed_data: &[u32], 
    length: usize
) -> Vec<u32> {
    
    // 1. Train the Model (Build Transition Table)
    // Map: TokenID -> List of all tokens that have ever followed it
    let mut transitions: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    
    for window in compressed_data.windows(2) {
        let current = window[0];
        let next = window[1];
        transitions.entry(current).or_default().push(next);
    }

    // 2. Generate
    let mut rng = rand::thread_rng();
    let mut output = Vec::with_capacity(length);
    
    // Start with a random token from the dataset
    let start_idx = rng.gen_range(0..compressed_data.len());
    let mut current_token = compressed_data[start_idx];
    output.push(current_token);

    for _ in 0..length {
        match transitions.get(&current_token) {
            Some(options) => {
                // simple unweighted choice from the history (correctly weighted by frequency)
                let next_token = options[rng.gen_range(0..options.len())];
                output.push(next_token);
                current_token = next_token;
            },
            None => break, // Dead end (end of song)
        }
    }
    output
}

pub fn save_vocab_to_bin(
    vocab: &Vec<[u8; CHUNK_SIZE]>,
    merge_rules: &Vec<(u32, u32, u32)>,
    path: &str
) -> io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);

    let base_vocab_size = vocab.len() as u32;
    let num_merges = merge_rules.len() as u32;

    // Header
    file.write_all(&base_vocab_size.to_le_bytes())?;
    file.write_all(&num_merges.to_le_bytes())?;

    // Base Tokens
    for token in vocab {
        file.write_all(token)?;
    }

    // Merge Rules
    for &(_, left, right) in merge_rules {
        file.write_all(&left.to_le_bytes())?;
        file.write_all(&right.to_le_bytes())?;
    }

    Ok(())
}

pub fn save_tokens_to_bin(
    tokens: &[u32],
    path: &str
) -> io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for &token in tokens {
        file.write_all(&token.to_le_bytes())?;
    }
    Ok(())
}

pub fn load_vocab(path: &str) -> io::Result<(Vec<[u8; CHUNK_SIZE]>, Vec<(u32, u32, u32)>)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let base_vocab_size = u32::from_le_bytes(buf4) as usize;

    reader.read_exact(&mut buf4)?;
    let num_merges = u32::from_le_bytes(buf4) as usize;

    let mut vocab = Vec::with_capacity(base_vocab_size);
    for _ in 0..base_vocab_size {
        let mut chunk = [0u8; CHUNK_SIZE];
        reader.read_exact(&mut chunk)?;
        vocab.push(chunk);
    }

    let mut merge_rules = Vec::with_capacity(num_merges);
    for i in 0..num_merges {
        let new_token_id = (base_vocab_size + i) as u32;
        reader.read_exact(&mut buf4)?;
        let left = u32::from_le_bytes(buf4);
        reader.read_exact(&mut buf4)?;
        let right = u32::from_le_bytes(buf4);
        merge_rules.push((new_token_id, left, right));
    }

    Ok((vocab, merge_rules))
}

pub fn load_sequence(path: &str) -> io::Result<Vec<u32>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut output = Vec::new();
    let mut buf = [0u8; 4];
    while let Ok(_) = reader.read_exact(&mut buf) {
        output.push(u32::from_le_bytes(buf));
    }
    Ok(output)
}
