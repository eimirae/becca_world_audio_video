import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import struct
import core.tools as tools
import becca_tools_control_panel.control_panel as cp
import os

import worlds.world_tools as wtools

class VideoVis:
    def __init__(self, TEST, fov_vert_span, fov_horz_span, num_video_sensors, VISUALIZE_PERIOD, frames_per_time_step, print_all_features):
        self.TEST = TEST
        self.fov_vert_span = fov_vert_span
        self.fov_horz_span = fov_horz_span
        self.num_video_sensors = num_video_sensors
        self.VISUALIZE_PERIOD = VISUALIZE_PERIOD
        self.frames_per_time_step = frames_per_time_step
        self.print_all_features = print_all_features
        
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
    
    def visualize(self, agent, timestep, frame_counter, video_sensors, intensity_image, clip_frame):
        """ Update the display to the user of the world's internal state """
        self.timestep = timestep
        self.frame_counter = frame_counter
        self.video_sensors = video_sensors
        self.intensity_image = intensity_image
        self.clip_frame = clip_frame
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
    
    