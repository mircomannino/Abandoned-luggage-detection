'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''

import cv2
from alert_boxByShape import *
from alert_configuration import AlertConfiguration
import time
from utils import create_parser
import os

if __name__ == '__main__':
    # Get arguments from command line
    args = create_parser()
    print('\n*************** Parsed arguments **************')
    for arg_name, arg_val in args.__dict__.items():
        print('\t', arg_name, ':', arg_val)
    print('***********************************************\n')

    # Check on video input
    if(not os.path.exists(args.input_file)):
        print('Error: {} does not exist'.format(args.input_file))
        exit()

    # Get info about the input video
    video_path = args.input_file
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    size = (width, height)

    # Create Alert configuration
    alert_cfg = AlertConfiguration()
    print(alert_cfg)

    ############################################################################
    print('*******************************************************************')
    # Frame to skip
    skipped_frames = alert_cfg.SKIPPED_FRAMES
    fps    = int(video.get(cv2.CAP_PROP_FPS) / skipped_frames)
    print('dimension: ', width, height)
    print('Actual fps: ', fps)
    print('Real fps: ', fps*skipped_frames)

    # Setup the output Video
    output_name_file = os.path.splitext(os.path.basename(args.input_file))[0] + '_outputBoxByShape' +'.mp4'
    output_path = os.path.join(args.output_dir, output_name_file )
    output_video = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    print('output_file: ', output_path)
    print('*******************************************************************')
    ############################################################################

    # Create the Alert object
    alert = Alert_BoxByShape(height=height, width=width, fps=fps, alert_cfg=alert_cfg)

    # Run the alarm system
    frame_counter = 0
    total_time = 0.0
    while(video.isOpened()):
        # Check if the frame is not null
        ret, frame = video.read()
        if not ret:
            break
        # skip some frame
        if(frame_counter % skipped_frames == 0):
            # Get initial time
            start_time_frame = time.time()

            # Resize frame
            frame = cv2.resize(frame, (width, height))

            # Update the Alert object
            alert.notify_alert(frame)

            # Update time information
            end_time_frame = time.time() - start_time_frame
            print("Frame: {0} - time: {1:.3f} s".format(frame_counter, end_time_frame))
            total_time += end_time_frame

            # Add frame to the output video
            image = cv2.resize(frame.astype('uint8'), size, interpolation = cv2.INTER_AREA)
            output_video.write(image)
            print('added: ', frame_counter)

        frame_counter += 1

    print('total time: ', total_time)

    output_video.release()
    print('\nDone! Output video in: ', output_path)
