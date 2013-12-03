import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import struct
import wave

from worlds.base_world import World as BaseWorld
import core.tools as tools
import becca_tools_control_panel.control_panel as cp
import worlds.world_tools as wtools
import becca_world_audio_video.audioVis as audioVis
import becca_world_audio_video.videoVis as videoVis

class World(BaseWorld):
    """ The watch world provides a sequence of video frames to the BECCA agent
    There are no actions that the agent can take that affect the world. 

    This world uses the OpenCV library. Installation instructions are here:
    http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation
    """
    # This package assumes that it is located directly under the BECCA package 
    def __init__(self, lifespan=None, test=False, visualize_period= 10 ** 4, print_all_features = True, fov_horz_span = 16, fov_vert_span = 12, name = 'audio-video_world', plot_all_features = False):
        super(World, self).__init__()
        if lifespan is not None:
            self.LIFESPAN = lifespan
        self.TEST = test
        self.VISUALIZE_PERIOD = visualize_period
        self.print_all_features = print_all_features
        self.fov_horz_span = fov_horz_span
        self.fov_vert_span = fov_vert_span
        self.name = name
        self.plot_all_features = plot_all_features

        self.DISPLAY_INTERVAL = visualize_period

        print "Entering", self.name
        
       
        self.frames_per_time_step = 3
        self.video_frame_rate = 29.947
       
        # Length of audio sample processed at each time step in ms
        self.snippet_length_ms = 1/self.video_frame_rate * 1000 / self.frames_per_time_step
        self.pad_length_ms = 100 * self.snippet_length_ms
        self.SAMPLING_FREQUENCY = 48000.
        self.snippet_length = int(np.floor(self.snippet_length_ms * 
                                           self.SAMPLING_FREQUENCY / 1000))
        self.pad_length = int(np.floor(self.pad_length_ms * 
                                       self.SAMPLING_FREQUENCY / 1000)) 
        # Step through the data such that each time step is synchronized with
        # a frame of video.
        # For actual speed video, use frames_per_time_step about equal to 
        # video_frame_rate * (self.snippet_length_ms / 2000)
        self.audio_samples_per_time_step = (
                self.frames_per_time_step * int(self.SAMPLING_FREQUENCY / 
                                           self.video_frame_rate))
       
       
        
        # Generate a list of the filenames to be used
        self.video_filenames = []
        extensions = ['.mpg', '.mp4', '.flv', '.avi']
        if self.TEST:
            test_filename = 'test_long'
            truth_filename = 'truth_long.txt'
            self.video_filenames = []
            self.audio_filenames = []
            self.video_filenames.append(os.path.join(
                    'becca_world_audio_video', 'test', test_filename + '.avi'))
            self.audio_filenames.append(os.path.join(
                    'becca_world_audio_video', 'test', test_filename + '.wav'))
            self.ground_truth_filename = os.path.join('becca_world_audio_video', 
                                                      'test', truth_filename)
            
        else:
            self.data_dir_name = os.path.join('becca_world_audio_video', 'data')
            self.video_filenames = tools.get_files_with_suffix(
                    self.data_dir_name, extensions)
            self.audio_filenames = tools.get_files_with_suffix(
                    self.data_dir_name, ['.wav'])
        self.video_file_count = len(self.video_filenames)
        self.audio_file_count = len(self.audio_filenames)
        
        
        print self.video_file_count, 'video files loaded.'
        print self.audio_file_count, 'audio files loaded.'
        
        
        # Initialize the data to be viewed
        current_file = self.initialize_video_file()
        self.initialize_audio_file(current_file)

        # Initialize the frequency bins
        self.frequencies = np.fft.fftfreq(self.snippet_length, 
                                d = 1/self.SAMPLING_FREQUENCY) 
        self.keeper_frequency_indices = np.where(self.frequencies > 0)
        self.frequencies = self.frequencies[self.keeper_frequency_indices] 
        tones_per_octave = 12.
        min_log2_freq = 5.
        num_octaves = 8.
        max_log2_freq = min_log2_freq + num_octaves
        num_bin_boundaries = num_octaves * tones_per_octave + 1
        self.bin_boundaries = np.logspace(min_log2_freq, max_log2_freq, 
                             num=num_bin_boundaries, endpoint=True, base=2.)
        self.bin_boundaries = np.concatenate((
                np.ones(1) * tools.EPSILON, self.bin_boundaries, 
                np.ones(1) * (np.amax(self.frequencies) + tools.EPSILON)))
        bin_membership = np.digitize(self.frequencies, self.bin_boundaries)
        self.bin_map = np.zeros((self.bin_boundaries.size - 1, 
                                self.frequencies.size))
        for bin_map_row in range(self.bin_map.shape[0]):
            self.bin_map[bin_map_row, 
                         np.where(bin_membership-1 == bin_map_row)] = 1.
        self.bin_map = self.bin_map / (np.sum(self.bin_map, axis=1) 
                                       [:,np.newaxis] + tools.EPSILON)
        # Hann window 
        self.window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(
                self.snippet_length) / (self.snippet_length - 1)))


        self.num_video_sensors = 2 * self.fov_horz_span * self.fov_vert_span
        self.num_audio_sensors = self.bin_map.shape[0]
        self.num_sensors = 2 * self.fov_horz_span * self.fov_vert_span + self.bin_map.shape[0]
        self.num_actions = 0
        self.video_panel = videoVis.VideoVis(self.TEST, self.fov_vert_span, self.fov_horz_span, 
                                             self.num_video_sensors, self.VISUALIZE_PERIOD, self.frames_per_time_step,
                                             self.print_all_features)
        self.audio_panel = audioVis.AudioVis(
                                        self.pad_length, 
                                        self.pad_length_ms, 
                                        self.snippet_length, 
                                        self.snippet_length_ms, 
                                        self.SAMPLING_FREQUENCY, 
                                        self.audio_samples_per_time_step, 
                                        self.bin_boundaries, 
                                        self.num_audio_sensors,
                                        self.name)
        self.frame_counter = 10000
        self.last_feature_visualized = 0

        
        if self.TEST:
            self.surprise_log_filename = os.path.join('', 
                                                      'log', 'surprise.txt')
            self.surprise_log = open(self.surprise_log_filename, 'w')


    def initialize_audio_file(self, filename_ext):
        """ Load an audio file and get ready to process it """
        filename, ext = os.path.splitext(filename_ext)
        filename = filename + '.wav'
        if not os.path.exists(filename):
            print "Failed loading ", filename
            return
        
        print 'Loading', filename
        self.audio_data = np.zeros(1)
        #TODO: Add try-catch for safe failure during file reading
        audio_file = wave.open(filename)
        audio_length = audio_file.getnframes()
        print audio_length, "audio length"
        sample_width = audio_file.getsampwidth()
        print sample_width, "sample width"
        frame_rate = audio_file.getframerate()
        print frame_rate, "frame rate"
        n_bits = 16.
        if frame_rate != self.SAMPLING_FREQUENCY:
            print 'Heads up: The audio file I just loaded has'
            print 'a frame rate of', frame_rate, ' but I\'m going'
            print 'to treat it as it if had a frame rate of'
            print self.SAMPLING_FREQUENCY
        self.audio_data = np.zeros(audio_length / sample_width)
        index = 0
        for i in range(audio_length / sample_width):
            wave_data = audio_file.readframes(1)
            # This is the magic incantation to get the data in the right format 
            data = struct.unpack("<h", wave_data)
            # Scale the audio signal by the maximum magnitude.
            # This normalizes the signal to fall on [-1, 1]
            self.audio_data[index] = data[0] / (2 ** (n_bits - 1))
            index += 1
        # Clean out any data points that read in funny or are non-numeric
        self.audio_data = np.delete(self.audio_data, 
                                    np.where(np.isnan(self.audio_data)), 0)
        self.padded_audio_data = np.concatenate((
                np.zeros(self.pad_length), 
                self.audio_data, np.zeros(self.pad_length)))
        # position_in_clip marks the point at the beginning of the snippet
        # that is being processed
        self.position_in_clip = 0
    


    def initialize_video_file(self):
        """ Queue up one of the video files and get it ready for processing """
        filename = self.video_filenames[
                np.random.randint(0, self.video_file_count)]
        print 'Loading', filename
        self.video_reader = cv2.VideoCapture(filename)
        self.clip_frame = 0
        return filename

    def step(self, action): 
        """ Advance the video one time step and read and process the frame """
        for _ in range(self.frames_per_time_step):
            ((success, image)) = self.video_reader.read() 
        # Check whether the end of the clip has been reached
        if not success:
            if self.TEST:
                # Terminate the test
                self.video_reader.release()
                self.surprise_log.close()
                print 'End of test reached'
                tools.report_roc(self.ground_truth_filename, 
                                 self.surprise_log_filename, self.name)
                sys.exit()
            else:
                self.initialize_video_file()
                ((success, image)) = self.video_reader.read() 
        self.timestep += 1
        self.clip_frame += self.frames_per_time_step
        self.position_in_clip += self.audio_samples_per_time_step 

        # Check whether the end of the clip has been reached
        if (self.position_in_clip + self.snippet_length + self.pad_length > 
            self.audio_data.size):
            if self.TEST:
                # Terminate the test if it's over
                self.surprise_log.close()
                print 'End of test reached'
                tools.report_roc(self.ground_truth_filename, 
                                self.surprise_log_filename, self.name)
                sys.exit()
            else:
                # If it's not, find another file 
                current_file = self.initialize_video_file()
                self.initialize_audio_file(current_file)
        # Generate a new audio snippet and set of sensor values
        self.snippet = self.audio_data[self.position_in_clip: 
                self.position_in_clip + self.snippet_length] * self.window

        magnitudes = np.abs(np.fft.fft(self.snippet).real) \
                            [self.keeper_frequency_indices]
        binned_magnitudes = np.dot(self.bin_map, magnitudes[:,np.newaxis])
        audio_sensors = np.log2(binned_magnitudes + 1.)


        image = image.astype('float') / 256.
        # Convert the color image to grayscale
        self.intensity_image = np.sum(image, axis=2) / 3.
        # Convert the grayscale to center-surround contrast pixels
        center_surround_pixels = wtools.center_surround(
                self.intensity_image, self.fov_horz_span, self.fov_vert_span)
        unsplit_sensors = center_surround_pixels.ravel()
        self.video_sensors = np.concatenate((np.maximum(unsplit_sensors, 0), 
                                       np.abs(np.minimum(unsplit_sensors, 0))))
        self.audio_sensors = audio_sensors
        self.sensors = np.append(self.audio_sensors, self.video_sensors)
        reward = 0
        print self.position_in_clip, "audio position"
        print self.clip_frame, "clip frame"
        return self.sensors, reward
        
    def set_agent_parameters(self, agent):
        """ Manually set some agent parameters, where required """
        agent.VISUALIZE_PERIOD = self.VISUALIZE_PERIOD
        if self.TEST:
            # Prevent the agent from adapting during testing
            agent.BACKUP_PERIOD = 10 ** 9
            for block in agent.blocks:
                block.ziptie.COACTIVITY_UPDATE_RATE = 0.
                block.ziptie.JOINING_THRESHOLD = 2.
                block.ziptie.AGGLOMERATION_ENERGY_RATE = 0.
                block.ziptie.NUCLEATION_ENERGY_RATE = 0.
                for cog in block.cogs:
                    cog.ziptie.COACTIVITY_UPDATE_RATE = 0.
                    cog.ziptie.JOINING_THRESHOLD = 2.
                    cog.ziptie.AGGLOMERATION_ENERGY_RATE = 0.
                    cog.ziptie.NUCLEATION_ENERGY_RATE = 0.
                    cog.daisychain.CHAIN_UPDATE_RATE = 0.
        else:
            pass
        return
    

    def visualize(self, agent):
        self.video_panel.visualize(agent, self.timestep, self.frame_counter, self.video_sensors, self.intensity_image, self.clip_frame)
        self.audio_panel.visualize(agent, self.TEST, self.snippet, self.position_in_clip, False, self.timestep, self.DISPLAY_INTERVAL, 
                              self.padded_audio_data, self.audio_sensors, self.frame_counter)
        self.frame_counter += 1

        return
    
    
    
