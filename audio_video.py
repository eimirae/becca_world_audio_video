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
        self.initialize_control_panel()
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
    
    def initialize_control_panel(self):
        """ Prepare the user display of the world's internal state """
        self.fig = cp.figure(number=2)
        self.ax_original_image = cp.subfigure(self.fig, 
                left=0., bottom=0.4, width=0.45, height=0.6)
        self.ax_sensed_image = cp.subfigure(self.fig, 
                left=0., bottom=0., width=0.3, height=0.4)
        self.ax_interpreted_image = cp.subfigure(self.fig, 
                left=0.3, bottom=0., width=0.3, height=0.4)
        self.ax_status = cp.subfigure(self.fig, 
                left=0.45, bottom=0.4, width=0.15, height=0.6)

        # Initialize original image 
        plt.gray()
        self.original_image = self.ax_original_image.imshow(
                np.zeros((self.fov_vert_span, self.fov_horz_span)), 
                vmin=0., vmax=1., interpolation='nearest', 
                animated=True)
        self.ax_original_image.text(-.01, -.01, 'Original image', 
                                    size=10, color=tools.OXIDE,
                                    ha='left', va='center')
        self.ax_original_image.get_xaxis().set_visible(False)
        self.ax_original_image.get_yaxis().set_visible(False)

        # Initialize sensed image
        plt.gray()
        self.sensed_image = self.ax_sensed_image.imshow(
                np.zeros((self.fov_vert_span, self.fov_horz_span)), 
                vmin=0., vmax=1., interpolation='nearest', 
                animated=True)
        self.ax_sensed_image.text(-.01, -.01, 'Sensed image', 
                                    size=10, color=tools.OXIDE,
                                    ha='left', va='center')
        self.ax_sensed_image.get_xaxis().set_visible(False)
        self.ax_sensed_image.get_yaxis().set_visible(False)

        # Initialize interpreted image
        plt.gray()
        self.interpreted_image = self.ax_interpreted_image.imshow(
                np.zeros((self.fov_vert_span, self.fov_horz_span)), 
                vmin=0., vmax=1., interpolation='nearest', 
                animated=True)
        self.ax_interpreted_image.text(-.01, -.01, 'Interpreted image', 
                                    size=10, color=tools.OXIDE,
                                    ha='left', va='center')
        self.ax_interpreted_image.get_xaxis().set_visible(False)
        self.ax_interpreted_image.get_yaxis().set_visible(False)

        # Initialize status window 
        self.ax_status.axis((0., 1., 0., 1.))
        self.ax_status.get_xaxis().set_visible(False)
        self.ax_status.get_yaxis().set_visible(False)
        self.clip_time_status = self.ax_status.text(-0.05, 0.13,
                    'Clip time:', 
                    color=tools.COPPER_SHADOW, size=10, ha='left', va='center')
        self.wake_time_status = self.ax_status.text(-0.05, 0.08,
                    'Wake time:', 
                    color=tools.COPPER_SHADOW, size=10, ha='left', va='center')
        self.life_time_status = self.ax_status.text(-0.05, 0.03,
                    'Life time:', 
                    color=tools.COPPER_SHADOW, size=10, ha='left', va='center')
        self.surprise_status = self.ax_status.text(-0.05, 0.4,
                    'Novelty: ', 
                    color=tools.COPPER_SHADOW, size=10, ha='left', va='center')

        # Initialize surprise plot 
        self.surprise_ax_left = 0.6
        self.surprise_ax_bottom = 0.56
        self.surprise_ax_width = 0.4
        self.surprise_ax_height = 0.44
        self.ax_surprise = cp.subfigure(self.fig, left=self.surprise_ax_left, 
                            bottom=self.surprise_ax_bottom, 
                            width=self.surprise_ax_width, 
                            height=self.surprise_ax_height)
        self.ax_surprise.axis((0., 1., 0., 1.))
        self.ax_surprise.get_xaxis().set_visible(False)
        self.ax_surprise.get_yaxis().set_visible(False)
        self.block_ax_vert_border = 0.02 * self.surprise_ax_height
        self.block_ax_horz_border = 0.04 * self.surprise_ax_width
        self.surprise_block_ax = []
        
        # Initialize features plot 
        self.feature_ax_left = 0.6
        self.feature_ax_bottom = 0.12
        self.feature_ax_width = 0.4
        self.feature_ax_height = 0.44
        self.ax_features = cp.subfigure(self.fig, left=self.feature_ax_left, 
                            bottom=self.feature_ax_bottom, 
                            width=self.feature_ax_width, 
                            height=self.feature_ax_height)
        self.ax_features.axis((0., 1., 0., 1.))
        self.ax_features.get_xaxis().set_visible(False)
        self.ax_features.get_yaxis().set_visible(False)
        self.feature_ax_vert_border = 0.028 * self.feature_ax_height
        self.feature_ax_horz_border = 0.005 * self.feature_ax_width
        self.block_ax = []
        self.fig.show()


    def visualize(self, agent):
        """ Update the display to the user of the world's internal state """
        if self.TEST:
            # Save the surprise value
            surprise_val = agent.surprise_history[-1]
            time_in_seconds = str(float(self.clip_frame) / 30.)
            file_line = ' '.join([str(surprise_val), str(time_in_seconds)])
            self.surprise_log.write(file_line)
            self.surprise_log.write('\n')
            self.audio_panel.visualize(agent, self.TEST, self.position_in_clip, self.surprise_log, self.timestep, self.DISPLAY_INTERVAL, 
                              self.padded_audio_data, self.audio_sensors)

        if (self.timestep % self.VISUALIZE_PERIOD != 0):
            return 
        print self.timestep, 'steps'
        (projections, feature_activities) = agent.get_projections()
        # Make a copy of projections for finding the interpretation
        interpretation_by_feature = list(projections)
        interpretation = np.zeros((self.num_video_sensors, 1))
        for block_index in range(len(interpretation_by_feature)):
            for feature_index in range(len(interpretation_by_feature
                                           [block_index])):
                this_feature_interpretation = (
                        interpretation_by_feature[block_index] 
                        [feature_index][:self.num_video_sensors,-1][:,np.newaxis])
                interpretation = np.maximum(interpretation, 
                        this_feature_interpretation *
                        feature_activities[block_index][feature_index])
        self.original_image.set_data(self.intensity_image)
        
        #import code
        #code.interact(local=locals())
        
        sensed_image_array = wtools.visualize_pixel_array_feature(
                self.video_sensors[:,np.newaxis], fov_horz_span=self.fov_horz_span,
                fov_vert_span=self.fov_vert_span, array_only=True) 
        self.sensed_image.set_data(sensed_image_array[0])
        interpreted_image_array = wtools.visualize_pixel_array_feature(
                interpretation[:self.num_video_sensors],  
                fov_horz_span=self.fov_horz_span,
                fov_vert_span=self.fov_vert_span, array_only=True) 
        self.interpreted_image.set_data(interpreted_image_array[0])
        # Update status window 
        self.clip_time_status.set_text(''.join((
                'Clip time: ', '%0.2f' % (self.clip_frame/(30.*60.)), ' min')))
        self.wake_time_status.set_text(''.join((
                'Wake time: ', '%0.2f' % (self.timestep * self.frames_per_time_step
                                           / (30.*60.)), ' min')))
        self.life_time_status.set_text(''.join((
                'Life time: ', '%0.2f' % (agent.timestep * self.frames_per_time_step
                                           / (30.*60.)), ' min')))
        self.surprise_status.set_text(''.join(( 
                'Novelty: ', '%0.2f' % agent.surprise_history[-1])))
        # Update surprise visualization window
        # Clear all axes
        for axes in self.surprise_block_ax:
            self.fig.delaxes(axes)
        self.surprise_block_ax = []
        # Display each block's features and bundle activities.
        # The top block has no bundles.
        num_blocks = len(agent.blocks)
        for block_index in range(num_blocks):
            block = agent.blocks[block_index]
            block_surprise = block.surprise
            num_cogs_in_block = len(block.cogs)
            surprise_array = np.reshape(block_surprise, 
                                        (num_cogs_in_block,
                                         block.max_bundles_per_cog)).T
            block_left = self.surprise_ax_left + self.block_ax_horz_border 
            block_height = ((self.surprise_ax_height -
                             self.block_ax_vert_border - 
                             self.feature_ax_vert_border * 2) / 
                             float(num_blocks) - 
                            self.block_ax_vert_border)
            block_bottom = (self.surprise_ax_bottom + 
                            self.feature_ax_vert_border +
                            self.block_ax_vert_border +
                            (block_height + self.block_ax_vert_border) * 
                            block_index)
            block_width = (self.surprise_ax_width - 
                           2 * self.block_ax_horz_border)
            last_block_top = block_bottom + block_height
            rect = (block_left, block_bottom, block_width, block_height)
            ax = self.fig.add_axes(rect, frame_on=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.gray()
            im = ax.imshow(surprise_array, aspect='auto', 
                           interpolation='nearest', vmin=0., vmax=1., 
                           cmap='copper')
            if block_index == 0:
                ax.text(num_cogs_in_block * 0.85,
                        block.max_bundles_per_cog * 0.8, 
                        'Novelty', color=tools.OXIDE, 
                        size=10, ha='left', va='bottom')
            self.surprise_block_ax.append(ax)

        # Update feature visualization window
        # Clear all axes
        for axes in self.block_ax:
            self.fig.delaxes(axes)
        self.block_ax = []
        # Display each block's features and bundle activities.
        # The top block has no bundles.
        num_blocks = len(agent.blocks)
        for block_index in range(num_blocks):
            block = agent.blocks[block_index]
            cable_activities = block.cable_activities
            num_cogs_in_block = len(block.cogs)
            activity_array = np.reshape(cable_activities, 
                                        (num_cogs_in_block,
                                         block.max_bundles_per_cog)).T
            block_left = self.feature_ax_left + self.block_ax_horz_border 
            block_height = ((self.feature_ax_height -
                             self.block_ax_vert_border - 
                             self.feature_ax_vert_border * 2) / 
                             float(num_blocks) - 
                            self.block_ax_vert_border)
            block_bottom = (self.feature_ax_bottom + 
                            self.feature_ax_vert_border +
                            self.block_ax_vert_border +
                            (block_height + self.block_ax_vert_border) * 
                            block_index)
            block_width = self.feature_ax_width - \
                            2 * self.block_ax_horz_border
            last_block_top = block_bottom + block_height
            rect = (block_left, block_bottom, block_width, block_height)
            ax = self.fig.add_axes(rect, frame_on=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.gray()
            im = ax.imshow(activity_array, aspect='auto', 
                           interpolation='nearest', vmin=0., vmax=1.,
                           cmap='copper')
            if block_index == 0:
                ax.text(num_cogs_in_block * 0.85,
                        block.max_bundles_per_cog * 0.8, 
                        'Activities', color=tools.OXIDE, 
                        size=10, ha='left', va='bottom')
            self.block_ax.append(ax)
        if self.print_all_features:
            log_directory = os.path.join('becca_world_audio_video', 'log')
            wtools.print_pixel_array_features(projections, self.num_video_sensors,
                                              self.num_actions, 
                                              self.fov_horz_span,
                                              self.fov_vert_span, 
                                              directory=log_directory,
                                              world_name=self.name)
        self.fig.canvas.draw()
        plt.draw()
        # Save the control panel image
        filename =  self.name + '_' + str(self.frame_counter) + '_video.png'
        full_filename = os.path.join('becca_world_audio_video', 'frames', filename)
        plt.figure(self.fig.number)
        #plt.savefig(full_filename, format='png', dpi=80) # for 720
        plt.savefig(full_filename, format='png', dpi=120) # for 1080
        
        self.audio_panel.visualize(agent, self.TEST, self.snippet, self.position_in_clip, False, self.timestep, self.DISPLAY_INTERVAL, 
                              self.padded_audio_data, self.audio_sensors, self.frame_counter)
        self.frame_counter += 1

        return
    
    
    
