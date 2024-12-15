import os
import random
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import fftconvolve
from tqdm import tqdm

# Define the base directories
base_ravdess_dir = ''
base_iemocap_dir = ''
base_noise_dir = ''
base_output_dir_ravdess = ''
base_output_dir_iemocap = ''

# Define noise types and SNR values
noise_types = ['babble', 'music', 'noise', 'white']
snr_values = [-5, 0, 5, 10, 15, 20]

# Define room sizes for reverberation
room_sizes = ['largeroom']


# Function to apply noise at a given SNR
def apply_noise(speech, noise_files, snr):
    while True:
        # Select a random noise file
        noise_file = random.choice(noise_files)
        noise = AudioSegment.from_wav(noise_file)
        
        # Repeat noise if it's shorter than the speech
        while len(noise) < len(speech):
            noise += noise
        
        # Trim noise to the same length as speech
        noise = noise[:len(speech)]
        
        # Calculate the power of the noise
        noise_power = noise.rms ** 2
        
        # If noise power is zero, choose another file
        if noise_power == 0:
            continue
        
        # Calculate the scaling factor to achieve the desired SNR
        speech_power = speech.rms ** 2
        target_noise_power = speech_power / (10 ** (snr / 10.0))
        scaling_factor = np.sqrt(target_noise_power / noise_power)
        noise = noise.apply_gain(20 * np.log10(scaling_factor))
        
        # Overlay noise on speech
        noisy_speech = speech.overlay(noise)
        return noisy_speech

# Function to convolve speech with RIR
def apply_reverb(speech, rir):
    speech_samples = np.array(speech.get_array_of_samples(), dtype=np.float32)
    rir_samples = np.array(rir.get_array_of_samples(), dtype=np.float32)

    # Scale the RIR before convolution to preserve reverberation effect
    rir_samples /= np.max(np.abs(rir_samples)) + 1e-7  # Normalize RIR to avoid clipping
    
    # Perform convolution
    reverberated_speech_samples = fftconvolve(speech_samples, rir_samples, mode='full')[:len(speech_samples)]
    
    # Apply soft clipping to preserve dynamics while avoiding distortion
    max_val = np.max(np.abs(reverberated_speech_samples)) + 1e-7
    scaling_factor = 0.9 / max_val
    reverberated_speech_samples = np.clip(reverberated_speech_samples * scaling_factor, -1.0, 1.0)
    
    # Convert back to int16 and create AudioSegment
    reverberated_speech = speech._spawn((reverberated_speech_samples * 32767).astype(np.int16).tobytes())
    
    # Optional: Apply a moderate gain to ensure audibility without overpowering the reverb
    reverberated_speech = reverberated_speech.apply_gain(3)  # Adjust the gain value as needed
    
    return reverberated_speech


# Function to process IEMOCAP dataset
def process_iemocap():
    for noise_type in tqdm(noise_types, desc="Processing noise types in IEMOCAP"):
        noise_files = [os.path.join(base_noise_dir, noise_type, 'train', f) for f in os.listdir(os.path.join(base_noise_dir, noise_type, 'train'))]
        for snr in tqdm(snr_values, desc=f"Processing SNR levels for {noise_type}", leave=False):
            for session in tqdm(os.listdir(base_iemocap_dir), desc="Processing sessions", leave=False):
                session_dir = os.path.join(base_iemocap_dir, session)
                
                # Process 'sentences' folder
                sentence_dir = os.path.join(session_dir, 'sentences', 'wav')
                if os.path.isdir(sentence_dir):
                    for subdir in os.listdir(sentence_dir):
                        subdir_path = os.path.join(sentence_dir, subdir)
                        if os.path.isdir(subdir_path):  # Ensure it's a directory
                            for file in os.listdir(subdir_path):
                                # Process only .wav files that don't start with a dot
                                if file.endswith('.wav') and not file.startswith('.'):
                                    speech_path = os.path.join(subdir_path, file)
                                    speech = AudioSegment.from_wav(speech_path)
                                    noisy_speech = apply_noise(speech, noise_files, snr)

                                    # Create the corresponding output directory maintaining the original structure
                                    output_subdir = os.path.join(base_output_dir_iemocap, noise_type, str(snr), session, 'sentences', 'wav', subdir)
                                    os.makedirs(output_subdir, exist_ok=True)

                                    # Save the noisy file to the corresponding directory
                                    output_path = os.path.join(output_subdir, file)
                                    noisy_speech.export(output_path, format='wav')

                # Process 'dialog' folder
                dialog_dir = os.path.join(session_dir, 'dialog', 'wav')
                if os.path.isdir(dialog_dir):
                    for file in os.listdir(dialog_dir):
                        # Process only .wav files that don't start with a dot
                        if file.endswith('.wav') and not file.startswith('.'):
                            speech_path = os.path.join(dialog_dir, file)
                            speech = AudioSegment.from_wav(speech_path)
                            noisy_speech = apply_noise(speech, noise_files, snr)

                            # Create the corresponding output directory maintaining the original structure
                            output_subdir = os.path.join(base_output_dir_iemocap, noise_type, str(snr), session, 'dialog', 'wav')
                            os.makedirs(output_subdir, exist_ok=True)

                            # Save the noisy file to the corresponding directory
                            output_path = os.path.join(output_subdir, file)
                            noisy_speech.export(output_path, format='wav')

    for room_size in tqdm(room_sizes, desc="Processing reverberation types in IEMOCAP"):
        room_dirs = [os.path.join(base_noise_dir, room_size, 'train', room) for room in os.listdir(os.path.join(base_noise_dir, room_size, 'train'))]
        for session in tqdm(os.listdir(base_iemocap_dir), desc="Processing sessions", leave=False):
            session_dir = os.path.join(base_iemocap_dir, session)
            
            # Process 'sentences' folder
            sentence_dir = os.path.join(session_dir, 'sentences', 'wav')
            if os.path.isdir(sentence_dir):
                for subdir in os.listdir(sentence_dir):
                    subdir_path = os.path.join(sentence_dir, subdir)
                    if os.path.isdir(subdir_path):  # Ensure it's a directory
                        for file in os.listdir(subdir_path):
                            # Process only .wav files that don't start with a dot
                            if file.endswith('.wav') and not file.startswith('.'):
                                speech_path = os.path.join(subdir_path, file)
                                speech = AudioSegment.from_wav(speech_path)
                                
                                # Randomly select a room and then an RIR file from that room
                                selected_room = random.choice(room_dirs)
                                rir_files = os.listdir(selected_room)
                                rir_file = random.choice(rir_files)
                                rir_path = os.path.join(selected_room, rir_file)
                                
                                rir = AudioSegment.from_wav(rir_path)
                                reverberated_speech = apply_reverb(speech, rir)

                                # Create the corresponding output directory maintaining the original structure
                                output_subdir = os.path.join(base_output_dir_iemocap, 'reverberation', room_size, session, 'sentences', 'wav', subdir)
                                os.makedirs(output_subdir, exist_ok=True)

                                # Save the reverberated file to the corresponding directory
                                output_path = os.path.join(output_subdir, file)
                                reverberated_speech.export(output_path, format='wav')

            # Process 'dialog' folder
            dialog_dir = os.path.join(session_dir, 'dialog', 'wav')
            if os.path.isdir(dialog_dir):
                for file in os.listdir(dialog_dir):
                    # Process only .wav files that don't start with a dot
                    if file.endswith('.wav') and not file.startswith('.'):
                        speech_path = os.path.join(dialog_dir, file)
                        speech = AudioSegment.from_wav(speech_path)
                        
                        # Randomly select a room and then an RIR file from that room
                        selected_room = random.choice(room_dirs)
                        rir_files = os.listdir(selected_room)
                        rir_file = random.choice(rir_files)
                        rir_path = os.path.join(selected_room, rir_file)
                        
                        rir = AudioSegment.from_wav(rir_path)
                        reverberated_speech = apply_reverb(speech, rir)

                        # Create the corresponding output directory maintaining the original structure
                        output_subdir = os.path.join(base_output_dir_iemocap, 'reverberation', room_size, session, 'dialog', 'wav')
                        os.makedirs(output_subdir, exist_ok=True)

                        # Save the reverberated file to the corresponding directory
                        output_path = os.path.join(output_subdir, file)
                        reverberated_speech.export(output_path, format='wav')


# Generate the noisy and reverberated versions for IEMOCAP
process_iemocap()

print("Noisy and reverberated versions of RAVDESS and IEMOCAP datasets have been generated.")
