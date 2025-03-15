import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import tkinter as tk
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams
from threading import Thread
import mne
from datetime import datetime
import os
import glob
from PIL import Image

class OddballExperiment:
    def __init__(self):
        self.target_pin = ''.join([str(random.randint(0, 9)) for _ in range(4)])
        self.data = {
            'eeg': [],
            'timestamps': [],
            'markers': [],
            'marker_timestamps': []
        }
        self.is_running = False
        self.subject_id = f"sub_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.num_trials = 100
        self.target_probability = 0.2  # 20% targets
        
        # Create marker stream
        info = StreamInfo('P300Markers', 'Markers', 1, 0, 'string', f'oddball_{self.subject_id}')
        self.marker_outlet = StreamOutlet(info)
        
        # Setup recording parameters
        self.stim_duration = 0.5  # seconds
        self.isi = 0.5  # inter-stimulus interval
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("P300 Oddball Paradigm")
        self.root.geometry("800x600")
        
        # Instructions frame
        instruction_frame = tk.Frame(self.root)
        instruction_frame.pack(pady=20)
        
        instruction_text = (
            "P300 Oddball Experiment\n\n"
            f"Target PIN: {self.target_pin}\n"
            "This is the PIN you should pay attention to.\n\n"
            "Instructions:\n"
            "1. Connect your EEG device and ensure it's streaming\n"
            "2. Click 'Start Experiment' when ready\n"
            "3. Focus on the screen and mentally count each time you see the target PIN"
        )
        
        tk.Label(instruction_frame, text=instruction_text, font=("Arial", 14)).pack()
        
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=20)
        
        self.status_var = tk.StringVar(value="Ready to start")
        tk.Label(control_frame, textvariable=self.status_var, font=("Arial", 12)).pack(pady=10)
        
        self.start_button = tk.Button(control_frame, text="Start Experiment", 
                                     command=self.start_experiment, font=("Arial", 14))
        self.start_button.pack(pady=10)
        
        # Stimulus display
        self.stim_frame = tk.Frame(self.root)
        self.stim_frame.pack(expand=True, fill="both")
        
        self.stim_label = tk.Label(self.stim_frame, text="", font=("Arial", 72))
        self.stim_label.pack(expand=True)
        
    def start_experiment(self):
        # Find EEG stream
        eeg_streams = [stream for stream in resolve_streams() if stream.type() == 'EEG']
        
        if not eeg_streams:
            self.status_var.set("ERROR: No EEG stream found. Please check your device.")
            return
            
        self.eeg_inlet = StreamInlet(eeg_streams[0])
        self.start_button.config(state="disabled")
        self.status_var.set(f"Experiment running: Target PIN is {self.target_pin}")
        
        # Start EEG recording in a separate thread
        self.is_running = True
        self.recording_thread = Thread(target=self.record_eeg)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start stimulus presentation
        self.present_stimuli()
    
    def present_stimuli(self):
        self.trial_count = 0
        self.root.after(1000, self.show_next_stimulus)  # Start after 1 second
    
    def show_next_stimulus(self):
        if self.trial_count >= self.num_trials or not self.is_running:
            self.finish_experiment()
            return
        
        self.trial_count += 1
        
        # Decide if this is a target trial
        is_target = random.random() < self.target_probability
        
        if is_target:
            pin = self.target_pin
            marker = "target"
        else:
            # Generate a different PIN
            while True:
                pin = ''.join([str(random.randint(0, 9)) for _ in range(4)])
                if pin != self.target_pin:
                    break
            marker = "non-target"
        
        # Display stimulus
        self.stim_label.config(text=pin)
        
        # Send marker
        self.marker_outlet.push_sample([marker])
        marker_time = time.time()
        self.data['markers'].append(marker)
        self.data['marker_timestamps'].append(marker_time)
        
        # Schedule stimulus removal
        self.root.after(int(self.stim_duration * 1000), 
                        lambda: self.stim_label.config(text=""))
        
        # Schedule next stimulus
        total_interval = int((self.stim_duration + self.isi) * 1000)
        self.root.after(total_interval, self.show_next_stimulus)
        
        # Update status
        self.status_var.set(f"Trial {self.trial_count}/{self.num_trials}")
    
    def record_eeg(self):
        while self.is_running:
            sample, timestamp = self.eeg_inlet.pull_sample(timeout=0.01)
            if sample:
                self.data['eeg'].append(sample)
                self.data['timestamps'].append(timestamp)
    
    def finish_experiment(self):
        self.is_running = False
        self.status_var.set("Experiment complete. Processing data...")
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=1.0)
        
        # Process and save data
        try:
            self.process_data()
        except Exception as e:
            import traceback
            print("Error processing data:", str(e))
            print(traceback.format_exc())
            self.status_var.set(f"Error processing data: {str(e)}")
        
        self.start_button.config(state="normal", text="Exit", command=self.root.destroy)
    
    def process_data(self):
        if not self.data['eeg']:
            self.status_var.set("No EEG data was collected!")
            print("No EEG data was collected!")
            return
            
        print(f"Processing data with {len(self.data['eeg'])} EEG samples and {len(self.data['markers'])} markers")
        
        # Convert EEG data to DataFrame
        eeg_df = pd.DataFrame(self.data['eeg'])
        eeg_df['timestamp'] = self.data['timestamps']
        
        # Create events DataFrame
        events_df = pd.DataFrame({
            'type': self.data['markers'],
            'timestamp': self.data['marker_timestamps']
        })
        
        # Save raw data
        eeg_filename = f"{self.subject_id}_eeg_data.csv"
        events_filename = f"{self.subject_id}_events.csv"
        
        eeg_df.to_csv(eeg_filename, index=False)
        events_df.to_csv(events_filename, index=False)
        print(f"Saved data to {eeg_filename} and {events_filename}")
        
        # Always show results window, even if analysis fails
        try:
            p300_filename = self.analyze_p300(eeg_df, events_df)
            print(f"P300 analysis completed, saved to {p300_filename}")
        except Exception as e:
            import traceback
            print("P300 analysis error:", str(e))
            print(traceback.format_exc())
            p300_filename = None
            self.status_var.set(f"Analysis error: {str(e)}")
        
        # Show results window regardless of P300 analysis success
        self.show_results(p300_filename)
    
    def show_results(self, p300_filename=None):
        """Display the results of the analysis"""
        if p300_filename is None:
            # Try to find latest P300 file
            p300_files = glob.glob("sub_*_p300.png")
            if p300_files:
                p300_filename = max(p300_files, key=os.path.getctime)
            else:
                print("No P300 files found!")
                return
        
        # Make sure we're looking for the PNG file
        if not p300_filename.endswith('.png'):
            p300_filename = f"{p300_filename}.png"
        
        # Check if the file exists
        if not os.path.exists(p300_filename):
            print(f"Error displaying plot: [Errno 2] No such file or directory: '{os.path.abspath(p300_filename)}'")
            return
        
        # Show the image
        img = Image.open(p300_filename)
        img.show()
        print(f"Displaying results from {p300_filename}")
    
    def analyze_p300(self, eeg_df, events_df):
        # Simple epoching and averaging
        sample_rate = 250  # Assumption - should be extracted from LSL metadata
        
        # Time window for epochs (in seconds)
        tmin = -0.2  # 200ms before stimulus
        tmax = 0.8   # 800ms after stimulus
        
        # Calculate window size in samples
        window_samples = int((tmax - tmin) * sample_rate)  # Should be 250 samples for default values
        
        # Create time points for plotting (time in seconds relative to stimulus)
        times = np.linspace(tmin, tmax, window_samples)
        
        # Create epochs
        epochs_target = []
        epochs_nontarget = []
        
        # Filter events
        target_events = events_df[events_df['type'] == 'target']
        nontarget_events = events_df[events_df['type'] == 'non-target']
        
        print(f"Found {len(target_events)} target events and {len(nontarget_events)} non-target events")
        
        # Validate we have data to work with
        if len(events_df) == 0 or len(eeg_df) == 0:
            print("ERROR: Empty EEG or events data")
            return None
        
        # Get time ranges
        eeg_start_time = eeg_df['timestamp'].iloc[0]
        eeg_end_time = eeg_df['timestamp'].iloc[-1]
        events_start_time = events_df['timestamp'].iloc[0]
        events_end_time = events_df['timestamp'].iloc[-1]
        
        eeg_duration = eeg_end_time - eeg_start_time
        events_duration = events_end_time - events_start_time
        
        print(f"EEG start time: {eeg_start_time}, Events start time: {events_start_time}")
        print(f"EEG timestamp range: {eeg_start_time} to {eeg_end_time}")
        print(f"Events timestamp range: {events_start_time} to {events_end_time}")
        print(f"EEG duration: {eeg_duration:.2f} seconds")
        print(f"Events duration: {events_duration:.2f} seconds")
        
        # Process both types of events
        for i, event in enumerate(target_events.itertuples()):
            event_time_ratio = (event.timestamp - events_start_time) / events_duration
            eeg_idx = int(len(eeg_df) * event_time_ratio)
            
            if i % 5 == 0:  # Only print some events to reduce output
                print(f"Target event {i}: time ratio {event_time_ratio:.3f}, mapped to EEG idx {eeg_idx}")
            
            # Calculate window around the stimulus
            start_idx = eeg_idx - int(sample_rate * abs(tmin))
            end_idx = eeg_idx + int(sample_rate * tmax)
            
            print(f"  -> start_idx={start_idx}, end_idx={end_idx}, eeg_length={len(eeg_df)}")
            
            # Check if we have enough data for this epoch
            if start_idx >= 0 and end_idx < len(eeg_df):
                # Extract data for channels 0-4 (assuming these are the EEG channels)
                # Note: We store each epoch as (window_samples, channels) for proper averaging
                epoch_data = eeg_df.iloc[start_idx:end_idx, 0:5].values
                epochs_target.append(epoch_data)
            else:
                # Skip this epoch if it's out of bounds
                print(f"Skipping target epoch {i} due to bounds: {start_idx}:{end_idx}")
        
        for i, event in enumerate(nontarget_events.itertuples()):
            event_time_ratio = (event.timestamp - events_start_time) / events_duration
            eeg_idx = int(len(eeg_df) * event_time_ratio)
            
            if i % 20 == 0:  # Only print some events to reduce output
                print(f"Non-target event {i}: time ratio {event_time_ratio:.3f}, mapped to EEG idx {eeg_idx}")
            
            # Calculate window around the stimulus
            start_idx = eeg_idx - int(sample_rate * abs(tmin))
            end_idx = eeg_idx + int(sample_rate * tmax)
            
            if i % 20 == 0:
                print(f"  -> start_idx={start_idx}, end_idx={end_idx}, eeg_length={len(eeg_df)}")
            
            # Check if we have enough data for this epoch
            if start_idx >= 0 and end_idx < len(eeg_df):
                # Extract data for channels 0-4 (assuming these are the EEG channels)
                # Note: We store each epoch as (window_samples, channels) for proper averaging
                epoch_data = eeg_df.iloc[start_idx:end_idx, 0:5].values
                epochs_nontarget.append(epoch_data)
            else:
                print(f"Skipping non-target epoch {i} due to bounds: {start_idx}:{end_idx}")
        
        print(f"Processed {len(epochs_target)} target epochs and {len(epochs_nontarget)} non-target epochs")
        
        if not epochs_target or not epochs_nontarget:
            print("No valid epochs found!")
            return None
        
        # Convert to numpy arrays and average
        try:
            # Convert list of epochs to numpy array for averaging
            epochs_target_np = np.array(epochs_target)
            epochs_nontarget_np = np.array(epochs_nontarget)
            
            # Average across epochs (first dimension)
            avg_target = np.mean(epochs_target_np, axis=0)
            avg_nontarget = np.mean(epochs_nontarget_np, axis=0)
            
            # Save results
            # Fix double "sub_" prefix by removing it from subject_id if present
            subject_id_clean = self.subject_id
            if subject_id_clean.startswith('sub_'):
                subject_id_clean = subject_id_clean[4:]
            
            p300_filename = f"sub_{subject_id_clean}_p300"
            # Save P300 data for later analysis
            np.savez(p300_filename, 
                    avg_target=avg_target, 
                    avg_nontarget=avg_nontarget,
                    times=times)
            
            # Plot the result
            self.plot_erp(avg_target, avg_nontarget, times, p300_filename)
            
            return p300_filename
            
        except Exception as e:
            print(f"P300 analysis error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_erp(self, avg_target, avg_nontarget, times, filename):
        plt.figure(figsize=(12, 8))
        
        # We have 5 channels, create a subplot for each
        for channel_idx in range(5):
            ax = plt.subplot(2, 3, channel_idx + 1)
            ax.plot(times, avg_target[:, channel_idx], 'r-', linewidth=2, label='Target')
            ax.plot(times, avg_nontarget[:, channel_idx], 'b-', linewidth=2, label='Non-Target')
            ax.axvline(x=0, linestyle='--', color='k', linewidth=1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (µV)')
            ax.set_title(f'Channel {channel_idx+1}')
            ax.legend()
        
        # P300 difference wave (Target - Non-Target) in last subplot
        ax = plt.subplot(2, 3, 6)
        for channel_idx in range(5):
            diff_wave = avg_target[:, channel_idx] - avg_nontarget[:, channel_idx]
            ax.plot(times, diff_wave, linewidth=2, label=f'Ch{channel_idx+1}')
        
        ax.axvline(x=0, linestyle='--', color='k', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Difference (µV)')
        ax.set_title('P300 Effect (Target - Non-Target)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{filename}.png")
        plt.close()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    experiment = OddballExperiment()
    experiment.run()
