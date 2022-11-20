import argparse
import time
import cv2
from me import Me
from me_line import MeLine


def count_frame(start_time, end_time, total_fps, frame_count):
    # count and calculate fps
    if end_time - start_time != 0:
        fps = 1 / (end_time - start_time)
        total_fps += fps
    else:
        print(f"cannot detect time, frame: {frame_count}")
        frame_count -= 1

    #print(frame_count)
    frame_count += 1
    return total_fps, frame_count


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v','--video',required=True,help='path to video')
    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(args['video'])
    if not cap.isOpened():
        print('Error while reading the video. Pls check if th path exists.')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_width)
    print(frame_height)

    out = cv2.VideoWriter('out.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (frame_width, frame_height))

    frame_count = 0
    total_fps = 0

    # singleton me
    me = Me()
    my_line = MeLine(me)

    while cap.isOpened():
        # capture each frame
        ret, frame = cap.read()
        if ret and frame_count < float("inf"):
            start_time = time.time()
            # rotate for being more similar to real scene
            #frame = cv2.rotate(frame, cv2.ROTATE_180)
            if frame_count == 0:

                my_line.init_handpick_track(8., frame)
                frame = my_line.display(frame)

            else:
                # TODO: ME core algorithm with old R, t
                my_line.track(frame)
                frame = my_line.display(frame)

            cv2.imshow('me_sample', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            end_time = time.time()
            total_fps, frame_count = count_frame(start_time, end_time, total_fps, frame_count)
            out.write(frame)

        else:
            break

    # release VideoCapture()
    cap.release()
    cv2.destroyAllWindows()

    # calculate average fps for lm optimization
    avg_fps = total_fps / frame_count if frame_count else "not defined"
    print(f"Average FPS: {avg_fps:.3f}")