import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from worlds.base_world import World as BaseWorld
import core.tools as tools
import becca_tools_control_panel.control_panel as cp
import becca_world_listen as listen
import worlds.world_tools as wtools

class World(BaseWorld):
    """ The watch world provides a sequence of video frames to the BECCA agent
    There are no actions that the agent can take that affect the world. 

    This world uses the OpenCV library. Installation instructions are here:
    http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation
    """
    # This package assumes that it is located directly under the BECCA package 
    def __init__(self, lifespan=None):
        super(World, self).__init__()
        if lifespan is not None:
            self.LIFESPAN = lifespan
        # Flag indicates whether the world is in testing mode
        self.short_test = False
        self.TEST = True
        self.VISUALIZE_PERIOD = 2 * 10 ** 3
        self.print_all_features = True
        self.fov_span = 10
        #self.name = 'watch_world'
        self.name = 'watch_world_bpc3'
        #self.name = 'watch_world_bpc10_long'
        print "Entering", self.name
        # Generate a list of the filenames to be used
        self.video_filenames = []
        extensions = ['.mpg', '.mp4', '.flv', '.avi']
        if self.TEST:
            if self.short_test:
                test_filename = 'test_short.avi'
                truth_filename = 'truth_short.txt'
            else:
                test_filename = 'test_long.avi'
                truth_filename = 'truth_long.txt'
            self.video_filenames = []
            self.video_filenames.append(os.path.join(
                    'becca_world_watch', 'test', test_filename))
            self.ground_truth_filename = os.path.join('becca_world_watch', 
                                                      'test', truth_filename)
        else:
            self.data_dir_name = os.path.join('becca_world_watch', 'data')
            self.video_filenames = tools.get_files_with_suffix(
                    self.data_dir_name, extensions)
        self.video_file_count = len(self.video_filenames)
        print self.video_file_count, 'video files loaded.'
        # Initialize the video data to be viewed
        self.initialize_video_file()

        self.VALUE_RANGE_DECAY_RATE = 10 ** -1
        self.BIN_RANGE_DECAY_RATE = 10 ** -4
        self.num_sensors = 2 * self.fov_span ** 2
        self.num_actions = 0
        self.initialize_control_panel()
        self.frame_counter = 10000
        self.frames_per_step = 3
        if self.TEST:
            self.surprise_log_filename = os.path.join('becca_world_watch', 
                                                      'log', 'surprise.txt')
            self.surprise_log = open(self.surprise_log_filename, 'w')

    def initialize_video_file(self):
        filename = self.video_filenames \
                [np.random.randint(0, self.video_file_count)]
        print 'Loading', filename
        self.video_reader = cv2.VideoCapture(filename)
        self.clip_frame = 0

    def step(self, action): 
        for _ in range(self.frames_per_step):
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
        self.clip_frame += self.frames_per_step
        image = image.astype('float') / 256.
        self.intensity_image = np.sum(image, axis=2) / 3.
        center_surround_pixels = wtools.center_surround(self.intensity_image,
                                                     self.fov_span)
        unsplit_sensors = center_surround_pixels.ravel()
        self.sensors = np.concatenate((np.maximum(unsplit_sensors, 0), 
                                       np.abs(np.minimum(unsplit_sensors, 0))))
        reward = 0
        return self.sensors, reward
        
    def set_agent_parameters(self, agent):
        agent.VISUALIZE_PERIOD = self.VISUALIZE_PERIOD
        if self.TEST:
            # Prevent the agent from adapting during testing
            agent.BACKUP_PERIOD = 10 ** 9
            for block in agent.blocks:
                block.ziptie.COACTIVITY_UPDATE_RATE = 0.
                block.ziptie.JOINING_THRESHOLD = 2.
                for cog in block.cogs:
                    cog.ziptie.COACTIVITY_UPDATE_RATE = 0.
                    cog.ziptie.JOINING_THRESHOLD = 2.
                    cog.daisychain.CHAIN_UPDATE_RATE = 0.
        else:
            pass
    
    def initialize_control_panel(self):
        self.fig = cp.figure()
        self.ax_original_image = cp.subfigure(self.fig, 
                left=0., bottom=0.4, width=0.35, height=0.6)
        self.ax_sensed_image = cp.subfigure(self.fig, 
                left=0.35, bottom=0.6, width=0.25, height=0.4)
        self.ax_interpreted_image = cp.subfigure(self.fig, 
                left=0.35, bottom=0.2, width=0.25, height=0.4)
        self.ax_status = cp.subfigure(self.fig, 
                left=0.35, bottom=0., width=0.4, height=0.2)
        self.ax_image_history = cp.subfigure(self.fig, 
                left=0., bottom=0., width=0.35, height=0.4)

        # Initialize original image 
        plt.gray()
        self.original_image = self.ax_original_image.imshow(
                np.zeros((10,10)), vmin=0., vmax=1., interpolation='nearest', 
                animated=True)
        self.ax_original_image.text(-1.2, -1., 'Original image', 
                                    size=10, color=tools.DARK_GREY,
                                    ha='left', va='center')
        self.ax_original_image.get_xaxis().set_visible(False)
        self.ax_original_image.get_yaxis().set_visible(False)

        # Initialize sensed image
        plt.gray()
        self.sensed_image = self.ax_sensed_image.imshow(
                np.zeros((10,10)), vmin=0., vmax=1., interpolation='nearest', 
                animated=True)
        self.ax_sensed_image.text(-1.2, -1., 'Sensed image', 
                                    size=10, color=tools.DARK_GREY,
                                    ha='left', va='center')
        self.ax_sensed_image.get_xaxis().set_visible(False)
        self.ax_sensed_image.get_yaxis().set_visible(False)


        # Initialize interpreted image
        plt.gray()
        self.interpreted_image = self.ax_interpreted_image.imshow(
                np.zeros((10,10)), vmin=0., vmax=1., interpolation='nearest', 
                animated=True)
        self.ax_interpreted_image.text(-1.2, -1., 'Interpreted image', 
                                    size=10, color=tools.DARK_GREY,
                                    ha='left', va='center')
        self.ax_interpreted_image.get_xaxis().set_visible(False)
        self.ax_interpreted_image.get_yaxis().set_visible(False)

        # Initialize image history

        # Initialize status window 
        self.ax_status.axis((0., 1., 0., 1.))
        self.ax_status.get_xaxis().set_visible(False)
        self.ax_status.get_yaxis().set_visible(False)
        self.age_status = self.ax_status.text(-0.05, 0.9,
                    'Clip time: 0 min', 
                    color=tools.DARK_GREY, size=10, ha='left', va='center')
        self.cumulative_age_status = self.ax_status.text(-0.05, 0.7,
                    'Total time: 0 min', 
                    color=tools.DARK_GREY, size=10, ha='left', va='center')
        self.surprise_status = self.ax_status.text(-0.05, 0.4,
                    'Novelty: ', 
                    color=tools.DARK_GREY, size=10, ha='left', va='center')

        # Initialize surprise plot 
        self.surprise_ax_left = 0.6
        self.surprise_ax_bottom = 0.6
        self.surprise_ax_width = 0.4
        self.surprise_ax_height = 0.4
        self.ax_surprise = cp.subfigure(self.fig, left=self.surprise_ax_left, 
                            bottom=self.surprise_ax_bottom, 
                            width=self.surprise_ax_width, 
                            height=self.surprise_ax_height)
        self.ax_surprise.axis((0., 1., 0., 1.))
        self.ax_surprise.get_xaxis().set_visible(False)
        self.ax_surprise.get_yaxis().set_visible(False)
        self.block_ax_vert_border = 0.01 * self.surprise_ax_height
        self.block_ax_horz_border = 0.04 * self.surprise_ax_width
        self.surprise_block_ax = []
        
        # Initialize features plot 
        self.feature_ax_left = 0.6
        self.feature_ax_bottom = 0.2
        self.feature_ax_width = 0.4
        self.feature_ax_height = 0.4
        self.ax_features = cp.subfigure(self.fig, left=self.feature_ax_left, 
                            bottom=self.feature_ax_bottom, 
                            width=self.feature_ax_width, 
                            height=self.feature_ax_height)
        self.ax_features.axis((0., 1., 0., 1.))
        self.ax_features.get_xaxis().set_visible(False)
        self.ax_features.get_yaxis().set_visible(False)
        self.feature_ax_vert_border = 0.025 * self.feature_ax_height
        self.feature_ax_horz_border = 0.005 * self.feature_ax_width
        self.block_ax = []
        self.fig.show()

    def visualize(self, agent):
        if self.TEST:
            # Save the surprise value
            surprise_val = agent.surprise_history[-1]
            time_in_seconds = str(float(self.clip_frame) / 30.)
            file_line = ' '.join([str(surprise_val), str(time_in_seconds)])
            self.surprise_log.write(file_line)
            self.surprise_log.write('\n')

        if (self.timestep % self.VISUALIZE_PERIOD != 0):
            return 
        print self.timestep, 'steps'
        (projections, feature_activities) = agent.get_projections()
        # Make a copy of projections for finding the interpretation
        interpretation_by_feature = list(projections)
        interpretation = np.zeros((self.num_sensors, 1))
        for block_index in range(len(interpretation_by_feature)):
            for feature_index in range(len(interpretation_by_feature
                                           [block_index])):
                this_feature_interpretation = (
                        interpretation_by_feature[block_index] 
                        [feature_index][:self.num_sensors,-1][:,np.newaxis])
                interpretation = np.maximum(interpretation, 
                        this_feature_interpretation *
                        feature_activities[block_index][feature_index])
        self.original_image.set_data(self.intensity_image)
        sensed_image_array = wtools.vizualize_pixel_array_feature(
                self.sensors[:,np.newaxis], array_only=True) 
        self.sensed_image.set_data(sensed_image_array[0])
        interpreted_image_array = wtools.vizualize_pixel_array_feature(
                interpretation[:self.num_sensors], array_only=True) 
        self.interpreted_image.set_data(interpreted_image_array[0])
        # Update status window 
        self.age_status.set_text(''.join((
                'Clip time: ', '%0.2f' % (self.clip_frame/(30.*60.)), ' min')))
        self.cumulative_age_status.set_text(''.join((
                'Total time: ', '%0.2f' % (self.timestep/(30.*60.)), ' min')))
        surprise = agent.surprise_history[-1]
        #surprise_mod = (np.log10(surprise + 1.) / 2.) - 1. 
        #surprise_mod = np.minimum(surprise_mod, 1.)
        surprise_mod = surprise
        self.surprise_status.set_text(''.join((
                'Novelty: ', '%0.2f' % surprise_mod)))
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
            block_surprise = block.surprise / 10.
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
            block_width = self.surprise_ax_width - \
                            2 * self.block_ax_horz_border
            last_block_top = block_bottom + block_height
            rect = (block_left, block_bottom, block_width, block_height)
            ax = self.fig.add_axes(rect, frame_on=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.gray()
            im = ax.imshow(surprise_array * 0.6 + 0.2, aspect='auto', 
                           interpolation='nearest', vmin=0., vmax=1.)
            if block_index == 0:
                ax.text(num_cogs_in_block * 0.85,
                        block.max_bundles_per_cog * 0.8, 
                        'Novelty', color=tools.LIGHT_GREY, 
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
            im = ax.imshow(activity_array * 0.6 + 0.2, aspect='auto', 
                           interpolation='nearest', vmin=0., vmax=1.)
            if block_index == 0:
                ax.text(num_cogs_in_block * 0.85,
                        block.max_bundles_per_cog * 0.8, 
                        'Activities', color=tools.LIGHT_GREY, 
                        size=10, ha='left', va='bottom')
            self.block_ax.append(ax)
        if self.print_all_features:
            log_directory = os.path.join('becca_world_watch', 'log')
            wtools.print_pixel_array_features(projections, self.num_sensors,
                                              self.num_actions, directory=log_directory,
                                              world_name='watch')
        self.fig.canvas.draw()
        plt.draw()
        # Save the control panel image
        filename =  self.name + '_' + str(self.frame_counter) + '.png'
        full_filename = os.path.join('becca_world_watch', 'frames', filename)
        self.frame_counter += 1
        plt.figure(self.fig.number)
        plt.savefig(full_filename, format='png', dpi=80) # for 720
        #plt.savefig(full_filename, format='png', dpi=120) # for 1080
        return
        
