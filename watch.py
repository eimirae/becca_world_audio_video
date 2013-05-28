import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from worlds.base_world import World as BaseWorld
import core.tools as tools
import becca_utils_control_panel.control_panel as cp

class World(BaseWorld):
    """ The listen world provides a stream of audio data to the BECCA agent
    There are no actions that the agent can take that affect the world. 
    """
    # This package assumes that it is located directly under the BECCA package 
    def __init__(self, lifespan=None):
        super(World, self).__init__()
        if lifespan is not None:
            self.LIFESPAN = lifespan
        # Flag indicates whether the world is in testing mode
        #self.TEST = True
        self.TEST = False 
        self.DISPLAY_INTERVAL = 10 ** 3
        self.name = 'listen_world'
        print "Entering", self.name
        self.sample_length_ms = 200
        pad_length_ms = 1000
        self.SAMPLING_FREQUENCY = 44100.
        self.pad_length = int(np.floor(pad_length_ms * 
                                       self.SAMPLING_FREQUENCY / 1000)) 
        self.snippet_length = int(np.floor(self.sample_length_ms * 
                                           self.SAMPLING_FREQUENCY / 1000))
        # Step through the data such that each time step is synchronized with
        # a frame of video
        frames_per_time_step = 3
        self.audio_samples_per_time_step = frames_per_time_step * \
                                        int(self.SAMPLING_FREQUENCY / 29.947)
        # Generate a list of the filenames to be used
        self.audio_filenames = []
        if self.TEST:
            filename = os.path.join('becca_world_listen', 'test', 'test.txt')
            self.audio_filenames.append(filename)
            self.ground_truth_filename = os.path.join('becca_world_listen', 
                                                      'test', 'truth.txt')
        else:
            self.data_dir_name = os.path.join('becca_world_listen', 'data')
            extensions = ['.txt']
            for localpath, directories, filenames in \
                        os.walk(self.data_dir_name):
                for filename in filenames:
                    for extension in extensions:
                        if filename.endswith(extension):
                            self.audio_filenames.append(os.path.join(localpath,
                                                                     filename))
        self.audio_file_count = len(self.audio_filenames)
        print self.audio_file_count, 'audio files loaded.'
        # Initialize the image_data to be viewed
        self.initialize_audio_file()

        self.frequencies = np.fft.fftfreq(self.snippet_length, 
                                d = 1/self.SAMPLING_FREQUENCY) 
        self.keeper_frequency_indices = np.where(self.frequencies > 0)
        self.frequencies = self.frequencies[self.keeper_frequency_indices] 
        tones_per_octave = 8.
        min_log2_freq = 6.
        num_octaves = 4.
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
        self.bin_map = self.bin_map / (np.sum(self.bin_map, axis=1) \
                                        [:,np.newaxis] + tools.EPSILON)
        # Hann window 
        self.window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(
                        self.snippet_length) / (self.snippet_length - 1)))
        self.min_val = tools.BIG
        self.max_val = -tools.BIG
        self.max_bin_vals = -tools.BIG * np.ones((self.bin_map.shape[0], 1)) 
        self.neutral_bin_vals = np.zeros((self.bin_map.shape[0], 1)) 
        self.min_bin_vals = tools.BIG * np.ones((self.bin_map.shape[0], 1))
        self.VALUE_RANGE_DECAY_RATE = 10 ** -1
        self.BIN_RANGE_DECAY_RATE = 10 ** -4
        self.MAX_NUM_FEATURES = 100       
        self.num_sensors = self.bin_map.shape[0]
        self.num_actions = 0
        self.initialize_control_panel()
        self.frame_counter = 10000
        self.last_feature_visualized = 0
        if self.TEST:
            self.surprise_log_filename = os.path.join('becca_world_listen', 
                                                      'log', 'surprise.txt')
            self.surprise_log = open(self.surprise_log_filename, 'w')

    def initialize_audio_file(self):
        filename = self.audio_filenames \
                [np.random.randint(0, self.audio_file_count)]
        print 'Loading', filename
        self.audio_data = np.loadtxt(filename)
        self.audio_data = np.delete(self.audio_data, 
                np.where(np.isnan(self.audio_data)), 0)
        self.padded_audio_data = np.concatenate((np.zeros(self.pad_length), 
                                self.audio_data, np.zeros(self.pad_length)))
        self.position_in_clip = 0
        if self.audio_data.size < self.snippet_length:
            print 'That clip was too short. Trying another.'
            self.initialize_audio_file()

    def step(self, action): 
        self.timestep += 1
        # Check whether the end of the clip has been reached
        if self.position_in_clip + 5 * self.snippet_length > self.audio_data.size:
            if self.TEST:
                # Terminate the test
                self.surprise_log.close()
                print 'End of test reached'
                self.report_roc()
                sys.exit()
            else:
                self.initialize_audio_file()
        # Generate a new audio snippet and set of sensor values
        self.snippet = self.audio_data[self.position_in_clip: 
                self.position_in_clip + self.snippet_length] * self.window
        magnitudes = np.abs(np.fft.fft(self.snippet).real) \
                            [self.keeper_frequency_indices]
        #self.sensors = np.dot(self.bin_map, magnitudes[:,np.newaxis])
        binned_magnitudes = np.dot(self.bin_map, magnitudes[:,np.newaxis])
        self.sensors = np.log2(binned_magnitudes + 1.)
        reward = 0
        self.position_in_clip += self.audio_samples_per_time_step 
        return self.sensors, reward
        
    def set_agent_parameters(self, agent):
        agent.VISUALIZE_PERIOD = 10 ** 3
        if self.TEST:
            # Prevent the agent from adapting during testing
            agent.BACKUP_PERIOD = 10 ** 9
            for block in agent.blocks:
                for cog in block.cogs:
                    cog.model.TRANSITION_UPDATE_RATE = 0.
                    cog.map.features_full = True
        else:
            pass
    
    def initialize_control_panel(self):
        self.dark_grey = (0.2, 0.2, 0.2)
        self.light_grey = (0.9, 0.9, 0.9)
        self.red = (0.9, 0.3, 0.3)
        self.fig = cp.figure()
        self.ax_snippet_long = cp.subfigure(self.fig, 
                left=0., bottom=0.65, width=0.6, height=0.35)
        self.ax_snippet = cp.subfigure(self.fig, 
                left=0., bottom=0.35, width=0.6, height=0.3)
        self.ax_status = cp.subfigure(self.fig, 
                left=0., bottom=0., width=0.2, height=0.35)
        self.ax_sensors = cp.subfigure(self.fig, 
                left=0.2, bottom=0., width=0.4, height=0.35)
        self.ax23 = cp.subfigure(self.fig, 
                left=0.6, bottom=0., width=0.25, height=0.1)

        # Initialize long snippet and surprise plot 
        self.long_snippet_length = self.pad_length * 2
        t_max = self.long_snippet_length * 1000 / self.SAMPLING_FREQUENCY
        self.time_steps_per_long_snippet = int(self.long_snippet_length / \
                self.audio_samples_per_time_step) 
        t_steps = np.linspace(0, t_max, self.time_steps_per_long_snippet)
        t = np.linspace(0, t_max, self.long_snippet_length)
        self.snippet_data_long, = self.ax_snippet_long.plot(t, 
                np.zeros((self.long_snippet_length)), color=self.dark_grey)
        min_x_limit = 0.
        max_x_limit = t_max
        self.ax_snippet_long.axis((min_x_limit, max_x_limit, -1., 1.))
        self.ax_snippet_long.add_patch(mpatches.Rectangle((900, -.99), 200, 
                1.98, facecolor=self.light_grey, edgecolor=self.dark_grey) )
        self.ax_snippet_long.text(min_x_limit +
                (max_x_limit - min_x_limit) * 0.05, 1.0, 'Audio data stream', 
                color=self.dark_grey, size=10, ha='left', va='bottom')
        self.ax_snippet_long.set_xlabel('time (ms)      .', 
                color=self.dark_grey, size=10, ha='right', va='center')
        self.ax_snippet_long.hold(True)
        self.surprise_data, = self.ax_snippet_long.plot(t_steps, -1 * np.ones((
                self.time_steps_per_long_snippet)), color=self.red,
                linewidth=2.)
        self.ax_snippet_long.text(min_x_limit +
                (max_x_limit - min_x_limit) * 0.05, -1.0, 'Novelty', 
                color=self.red, size=10, ha='left', va='bottom')
        
        # Initialize snippet plot 
        t_max = self.snippet_length * 1000 / self.SAMPLING_FREQUENCY
        t = np.linspace(0, t_max, self.snippet_length)
        self.snippet_data, = self.ax_snippet.plot(t, 
                np.zeros((self.snippet_length)), color=self.dark_grey)
        min_x_limit = 0.
        max_x_limit = t_max
        self.ax_snippet.axis((min_x_limit, max_x_limit, -1., 1.))
        self.ax_snippet.text(min_x_limit + (max_x_limit - min_x_limit) * 0.05, 
                1.0, 'Audio snippet', color=self.dark_grey, size=10,
                ha='left', va='bottom')
        self.ax_snippet.set_xlabel('time (ms)      .', color=self.dark_grey, 
                size=10, ha='right', va='center')
        
        # Initialize sensors window 
        self.bar_width = 0.01
        self.sensor_data = self.ax_sensors.barh(
                np.log10(self.bin_boundaries[1:]) - self.bar_width/2, 
                np.zeros(self.num_sensors), height=self.bar_width, 
                color=self.dark_grey)
        self.min_y_sensor_limit = np.log10(30)
        self.max_y_sensor_limit = np.log10(20000)
        self.ax_sensors.axis((0., 5., self.min_y_sensor_limit, 
                self.max_y_sensor_limit))
        self.fig.canvas.draw()
        labels = [item.get_text() for item 
                  in self.ax_sensors.get_yticklabels()]
        labels[0] = ''
        labels[1] = ''
        labels[2] = '100'
        labels[3] = ''
        labels[4] = '1000'
        labels[5] = ''
        labels[6] = '10000'

        self.ax_sensors.set_yticklabels(labels)
        self.ax_sensors.text(0.05, self.min_y_sensor_limit + 
                (self.max_y_sensor_limit - self.min_y_sensor_limit) * 0.94, 
                'Sensors', color=self.dark_grey, size=10, 
                ha='left', va='bottom')
        self.ax_sensors.set_ylabel('frequency (Hz)', color=self.dark_grey, 
                size=10, ha='right', va='center')
        # Initialize status window 
        self.ax_status.axis((0., 1., 0., 1.))
        self.ax_status.get_xaxis().set_visible(False)
        self.ax_status.get_yaxis().set_visible(False)
        self.age_status = self.ax_status.text(-0.05, 0.9,
                    'Clip time: 0 min', 
                    color=self.dark_grey, size=10, ha='left', va='center')
        self.cumulative_age_status = self.ax_status.text(-0.05, 0.8,
                    'Total time: 0 min', 
                    color=self.dark_grey, size=10, ha='left', va='center')
        
        # Initialize features plot 
        self.feature_ax_left = 0.6
        self.feature_ax_bottom = 0.1
        self.feature_ax_width = 0.4
        self.feature_ax_height = 0.9
        self.ax_features = cp.subfigure(self.fig, left=self.feature_ax_left, 
                            bottom=self.feature_ax_bottom, 
                            width=self.feature_ax_width, 
                            height=self.feature_ax_height)
        self.ax_features.axis((0., 1., 0., 1.))
        self.ax_features.get_xaxis().set_visible(False)
        self.ax_features.get_yaxis().set_visible(False)
        self.block_ax_vert_border = 0.04 * self.feature_ax_height
        self.block_ax_horz_border = 0.04 * self.feature_ax_width
        self.feature_ax_vert_border = 0.005 * self.feature_ax_height
        self.feature_ax_horz_border = 0.005 * self.feature_ax_width
        self.state_ax_vert_border = 0.005 * self.feature_ax_height
        self.state_ax_horz_border = 0.005 * self.feature_ax_width
        self.block_ax = []
        self.feature_ax = []
        self.state_ax = []
        
        # Initialize heartbeat plot         
        self.x = np.linspace(0, 6*np.pi, 100)
        self.line1, = self.ax23.plot(self.x, np.sin(self.x), 'k-')
        self.ax23.get_xaxis().set_visible(False)
        self.ax23.get_yaxis().set_visible(False)
        self.phase = 0.
        self.fig.show()

    def visualize(self, agent):
        if self.TEST:
            # Save the surprise value
            surprise_val = agent.surprise_history[-1]
            time_in_seconds = str(float(self.position_in_clip) / 
                                  float(self.SAMPLING_FREQUENCY))
            file_line = ' '.join([str(surprise_val), str(time_in_seconds)])
            self.surprise_log.write(file_line)
            self.surprise_log.write('\n')

        if (self.timestep % self.DISPLAY_INTERVAL != 0):
            return 
        # Update surprise data 
        half_length = int(self.time_steps_per_long_snippet/ 2)
        surprise = [0] * self.time_steps_per_long_snippet
        if len(agent.surprise_history) < half_length:
            surprise[half_length - len(agent.surprise_history): half_length] \
                    = agent.surprise_history
        else:
            surprise[:half_length] = agent.surprise_history[-half_length:]
        surprise_mod = (np.log10(np.asarray(surprise) + 1.) / 2.) - 1. 
        surprise_mod = np.minimum(surprise_mod, 1.)
        self.surprise_data.set_ydata(surprise_mod)
        # Update long snippet data
        start_sample = self.position_in_clip - self.pad_length * .9 + \
                                self.pad_length            
        long_snippet = self.padded_audio_data[start_sample: 
                                    start_sample + self.long_snippet_length]
        scale_factor = 0.5
        long_snippet = long_snippet / scale_factor 
        self.snippet_data_long.set_ydata(long_snippet)
        # Update snippet plot 
        self.snippet_data.set_ydata(self.snippet)
        # Update sensors window 
        for i in range(len(self.sensor_data)):
            self.sensor_data[i].set_width(self.sensors[i])
        # Update status window 
        self.age_status.set_text('Clip time: ' + 
                    str(float(int(100 * self.position_in_clip /
                    (self.SAMPLING_FREQUENCY * 60.)))/100.) + ' min')
        self.cumulative_age_status.set_text('Total time: ' + 
                    str(float(int(100 * self.timestep * 
                    self.audio_samples_per_time_step /
                    (self.SAMPLING_FREQUENCY * 60.)))/100.) + ' min')
        # Update feature visualization window
        # Clear all axes
        for axes in self.block_ax:
            self.fig.delaxes(axes)
        self.block_ax = []
        for axes in self.feature_ax:
            self.fig.delaxes(axes)
        self.feature_ax = []
        for axes in self.state_ax:
            self.fig.delaxes(axes)
        self.state_ax = []
        # How to display features of different blocks?
        projections = agent.get_projections()
        if len(projections) > 0:        
            num_cols = len(projections)
            total_rows = num_cols + 1
            # How many state columns
            while (num_cols < total_rows * 2):
                num_cols += 1
                total_rows = 0
                # Given N state columns, how many rows? 
                for block_index in range(len(projections)):
                    features_in_block = len(projections[block_index])
                    states_per_feature = block_index + 2
                    # automatically rounds down
                    features_per_row = num_cols / states_per_feature
                    total_rows += int(np.ceil(float(features_in_block) /
                                              float(features_per_row)))
            # Display each block's features
            last_block_top = self.feature_ax_bottom + \
                             self.block_ax_vert_border 
            for block_index in range(len(projections)):
                features_in_block = len(projections[block_index])
                states_per_feature = block_index + 2
                features_per_row = num_cols / states_per_feature
                num_rows = int(np.ceil(float(features_in_block) /
                                       float(features_per_row)))
                block_left = self.feature_ax_left + self.block_ax_horz_border 
                block_bottom = last_block_top 
                block_width = self.feature_ax_width - \
                                2 * self.block_ax_horz_border
                block_height = float(num_rows)/float(total_rows) * \
                        (self.feature_ax_height - 
                         2 * self.block_ax_vert_border)
                last_block_top = block_bottom + block_height
                rect = (block_left, block_bottom, block_width, block_height)
                ax = self.fig.add_axes(rect, frame_on=False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.axis((0., 1., 0., 1.))
                ax.plot(np.asarray((0., 0., 1., 1., 0.)), 
                        np.asarray((0., 1., 1., 0., 0.)), 
                        color=self.dark_grey)
                #ax.text(-0.02, 0.98,  str(block_index), 
                #        color=self.dark_grey, size=10, ha='left', va='top')
                self.block_ax.append(ax)
                feature_index = -1
                for row in np.arange(num_rows - 1, -1, -1):
                    for col in range(features_per_row):
                        feature_index += 1
                        feature_left =  float(col * states_per_feature) / \
                                float(features_per_row 
                                * states_per_feature) * \
                                block_width + block_left + \
                                self.feature_ax_horz_border 
                        feature_bottom = float(row)/float(num_rows) * \
                                block_height + block_bottom + \
                                self.feature_ax_vert_border
                        feature_width =  float(states_per_feature) / \
                                float(features_per_row * 
                                states_per_feature) * block_width - \
                                2 * self.feature_ax_horz_border 
                        feature_height =  1./float(num_rows) * block_height -\
                                2 * self.feature_ax_vert_border 
                        rect = (feature_left, feature_bottom, 
                                feature_width, feature_height)
                        ax = self.fig.add_axes(rect, frame_on=False)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax.axis((0., 1., 0., 1.))
                        ax.plot(np.asarray((0., 0., 1., 1., 0.)), 
                                np.asarray((0., 1., 1., 0., 0.)), 
                                color=self.dark_grey)
                        ax.text(0.1, 0.97, str(feature_index), 
                                color=self.dark_grey, size=10, 
                                ha='left', va='top')
                        self.feature_ax.append(ax)
                        if feature_index < features_in_block:
                            #debug
                            #feature_activity = (agent.blocks[block_index].
                            #                 feature_outputs[feature_index, 0])
                            feature_activity = 1.
                            for state in range(states_per_feature): 
                                left =  float(state) / \
                                        float(states_per_feature) * \
                                        feature_width + feature_left + \
                                        self.state_ax_horz_border
                                bottom = feature_bottom + \
                                        self.state_ax_vert_border
                                width =  feature_width/ \
                                        float(states_per_feature) - \
                                        2 * self.state_ax_horz_border 
                                height =  feature_height - \
                                        3 * self.state_ax_vert_border
                                rect = (left, bottom, width, height)
                                ax = self.fig.add_axes(rect, frame_on=False)
                                ax.get_xaxis().set_visible(False)
                                ax.get_yaxis().set_visible(False)
                                bar_centers = np.log10(
                                        self.bin_boundaries[1:]) - \
                                        self.bar_width/2 
                                bar_centers[-1] += self.bar_width
                                # The color of the bars reflects the 
                                # activity block of the feature
                                color_val = 0.8 * (1. - feature_activity)
                                bar_color = (color_val, color_val, color_val) 
                                ax.barh(bar_centers,  
                                        projections[block_index][feature_index]
                                        [self.num_actions:self.num_actions + 
                                         self.num_sensors, state], 
                                        height=self.bar_width, 
                                        color=bar_color, edgecolor=bar_color)
                                ax.plot(np.asarray((0.03, 0., 0., 0.03)), 
                                        np.asarray((2.05, 2.05, 2.95, 2.95)), 
                                        color=bar_color, linewidth=2)
                                ax.axis((0., 1., self.min_y_sensor_limit, 
                                         self.max_y_sensor_limit))
                                self.state_ax.append(ax)
                for plot_index in range(len(projections[block_index])):
                    feature_activity = 1.
                    plt.close(99)
                    feature_fig = plt.figure(num=99)
                    for state in range(states_per_feature):                
                        left =  float(state) / float(states_per_feature) 
                        bottom = 0.
                        width =  1. /  float(states_per_feature)
                        height =  1
                        rect = (left, bottom, width, height)
                        ax = feature_fig.add_axes(rect)
                        bar_centers = np.log10(
                                self.bin_boundaries[1:]) - \
                                self.bar_width/2 
                        bar_centers[-1] += self.bar_width
                        # The color of the bars reflects the 
                        # activity block of the feature
                        color_val = 0.8 * (1. - feature_activity)
                        bar_color = (color_val, color_val, color_val) 
                        ax.barh(bar_centers,  
                                projections[block_index][plot_index]
                                [self.num_actions:self.num_actions + 
                                 self.num_sensors, state], 
                                height=self.bar_width, 
                                color=bar_color, edgecolor=bar_color)
                        ax.plot(np.asarray((0.03, 0., 0., 0.03)), 
                                np.asarray((2.05, 2.05, 2.95, 2.95)), 
                                color=bar_color, linewidth=2)
                        ax.axis((0., 1., self.min_y_sensor_limit, 
                                 self.max_y_sensor_limit))
                    # create a plot of individual features
                    filename = '_'.join(('block', str(block_index),
                                         'feature', str(plot_index),
                                         'listen', 'world.png'))
                    full_filename = os.path.join('becca_world_listen',
                                                 'log', filename)
                    plt.title(filename)
                    plt.savefig(full_filename, format='png') 
        self.fig.canvas.draw()
        # Update heartbeat window
        self.phase += 0.1
        self.line1.set_ydata(np.sin(self.x + self.phase))
        plt.draw()
        # Save the control panel image
        filename =  self.name + '_' + str(self.frame_counter) + '.png'
        full_filename = os.path.join('becca_world_listen', 'frames', filename)
        self.frame_counter += 1
        plt.figure(self.fig.number)
        #plt.savefig(full_filename, format='png', dpi=80) # for 720
        plt.savefig(full_filename, format='png', dpi=120) # for 1080
        return
        
    def report_roc(self):
        """
        Report the Receiver Operating Characteristic curve

        Plot the true positive rate (the number of correctly identified
        targets divided by the total number of targets) against the
        false positive rate (the number of data points erroneously 
        identified as targets divided by the total number of 
        non-target data points).
        """
        truth = np.loadtxt(self.ground_truth_filename)
        surprise = np.loadtxt(self.surprise_log_filename)
        log_surprise = np.log10(surprise[:,0] + 1.)
        times = surprise[:,1]
        # If a target is identified within delta seconds, that is close enough
        delta = 1.
        starts = truth[:,0] - delta
        ends = truth[:,1] + delta
        total_num_targets = starts.size
        # Total up potential false positives.
        total_non_target_points = 0
        for time in times:
            after_start = np.where(time > starts, True, False)
            before_end = np.where(time < ends, True, False)
            target_match = np.logical_and(after_start, before_end)
            if not target_match.any():
                total_non_target_points += 1

        false_positive_rate = []
        true_positive_rate = []
        thresholds = np.linspace(0, np.max(log_surprise), num=100)
        for threshold in thresholds:
            # Determine the false positive rate, i.e. how many
            # of all possible false positives were reported
            above_threshold_indices = np.where(log_surprise > threshold)
            above_threshold_times = times[above_threshold_indices]
            num_false_positives = 0
            for time in above_threshold_times:
                after_start = np.where(time > starts, True, False)
                before_end = np.where(time < ends, True, False)
                target_match = np.logical_and(after_start, before_end)
                if not target_match.any():
                    num_false_positives += 1
            false_positive_rate.append(float(num_false_positives) / \
                                        (float(total_non_target_points) +
                                         tools.EPSILON))
            # Determine the true positive rate, i.e.
            # what fraction of the targets were identified 
            num_targets_identified = 0
            for indx in range(total_num_targets):
                after_start = np.where(times[above_threshold_indices] > 
                                       starts[indx], True, False)
                before_end = np.where(times[above_threshold_indices] < 
                                      ends[indx], True, False)
                target_match = np.logical_and(after_start, before_end)
                if target_match.any():
                    num_targets_identified += 1
            true_positive_rate.append(float(num_targets_identified)/
                                              (float(total_num_targets) + 
                                               tools.EPSILON))
        # Show surprise over time 
        fig = plt.figure(tools.str_to_int('surprise'))
        fig.clf()
        plt.plot(times, log_surprise)
        plt.title('Novel target identification signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Estimated novelty strength')
        plt.hold(True)
        ax = plt.gca()
        # Show the temporal locations of the targets
        for target_index in range(total_num_targets):
            ax.add_patch(mpatches.Rectangle(
                    (starts[target_index], 0.03), 
                    ends[target_index] - starts[target_index], 
                    np.max(log_surprise), 
                    facecolor=self.light_grey, edgecolor=self.dark_grey))
            
        # Show the ROC curve
        fig = plt.figure(tools.str_to_int('roc'))
        fig.clf()
        plt.plot(false_positive_rate, true_positive_rate)
        plt.title('Receiver operating characteristic (ROC) curve for audio')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.axis((-0.1, 1.1, -0.1, 1.1))
        plt.ioff()
        plt.show()    
