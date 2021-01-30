# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:39:39 2020

@author: Jason
"""


# Liquid crystal video phase labeller

# Given a model trained for LC texture phase classification
# this script will label a given video with the determined
# phase at each frame and output a phase prediction 
# confidence graph against temperature

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model
import cv2
import collections
import pandas as pd


class PhaseLabeller:
    
    def __init__(self):
        
        self.model = None
        # Initialize a dict which will be filled
        # with phase prediction confidences and temperatures
        self.confidence_dict = {"Temperature":[]}
        self.phase_list = []
        
        
    def get_model(self, model_load_dir, phase_list):
        # Load in the trained model
        self.model = tf.keras.models.load_model(model_load_dir)
        for layer in self.model.layers:
            layer.trainable = False
        self.model.summary()
        
        self.phase_list = phase_list
        # Create empty lists to store prediction accuracy for each phase
        for phase in phase_list:
            self.confidence_dict[phase] = []
        
        
        
    def label_video(self, vid_path, vid_save_path,
                    start_temp, end_temp=None,
                    temp_rate_per_sec=None):
        
        # Get the video to be labelled
        vid_stream = cv2.VideoCapture(vid_path)
        writer = None
        width = int(vid_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(vid_stream.get(cv2.CAP_PROP_FOURCC))

        # Relate frames of videos to certain temperature
        fps = vid_stream.get(cv2.CAP_PROP_FPS)
        frame_count = int(vid_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        #vid_duration = frame_count/fps
        if end_temp != None:
            temp_step_per_frame = (end_temp - start_temp)/frame_count
        else:
            temp_step_per_frame = temp_rate_per_sec/fps
        
        i = 0
        while True:
            (grabbed, frame) = vid_stream.read()
            if not grabbed:
                break
        
            output_vid = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frame = cv2.resize(frame, (256, 256)).astype("float32") / 255
            frame = cv2.resize(frame, (256, 256)).astype("float32")
    
            # Get phase prediction for frame (doesnt work for binary classifier right now)
            pred = self.model.predict(np.expand_dims(frame, axis=(0,3)))
            y_predict = np.argmax(pred, axis=1)
            confidence = pred[0][y_predict]
            for n, phase in enumerate(self.phase_list):
                # Store data for plotting
                self.confidence_dict[phase].append(pred[0][n])
            label = self.phase_list[int(y_predict)]
            
            # Get temperature of frame
            temp = start_temp + i*temp_step_per_frame
            self.confidence_dict["Temperature"].append(temp)
            i += 1
            
            phase_text = "Phase: {}".format(label)
            phase_confidence_text = "Confidence: {}".format(confidence)
            temp_text = "Temperature: %.1f C"%(temp)
            cv2.putText(output_vid, phase_text, (35,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (255, 255, 255), 5)
            cv2.putText(output_vid, phase_confidence_text, (35,100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (255, 255, 255), 5)
            cv2.putText(output_vid, temp_text, (35,150), cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (255, 255, 255), 5)
            
            if writer is None:
                writer = cv2.VideoWriter(vid_save_path, fourcc, fps, (width, height), True) 
            writer.write(output_vid)
            cv2.imshow("Output_vid", output_vid)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break       
        writer.release()
        vid_stream.release()
        cv2.destroyAllWindows()
        
        
        
    def to_csv(self, save_dir):
        temp_confidence_data = pd.DataFrame(data=self.confidence_dict,
                                            columns=self.phase_list)
        temp_confidence_data.head()
        temp_confidence_data.to_csv(save_dir)
        
    def conf_temp_plot(self, phase_plot_list, color_list, save_path,
                       x_tick_range=None, start_temp=None, end_temp=None):
        
        plt.figure(figsize=(15,10))
        temp = self.confidence_dict["Temperature"]
        for n, phase in enumerate(phase_plot_list):
            phase_conf = self.confidence_dict[phase]
            color = color_list[n]
            plt.scatter(temp, phase_conf, c=color, marker="x", label=phase)
        
        plt.xlabel("Temperature (Degrees Celsius)")
        if x_tick_range != None:
            plt.xticks(x_tick_range)
        if start_temp != None and end_temp != None:
            plt.xlim(start_temp, end_temp)
        plt.ylabel("Phase prediction confidence")
        plt.yticks(np.arange(0, 1.1, 0.1))
        
        plt.legend(loc="best")
        plt.grid(linestyle="--")
        
        plt.savefig(save_path)
        
        
        