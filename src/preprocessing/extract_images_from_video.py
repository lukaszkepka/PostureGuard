import argparse
import os

import cv2
import os.path as path

key_mapping = {
    'a': 'not_correct',
    'd': 'correct'
}


def parse_args():
    ap = argparse.ArgumentParser(description="Extracts and displays consecutive frames from video file. Each frame can "
                                             "be classified as 'correct' or 'not_correct' by pressing keys 'a' or 'd'. "
                                             "When frame is classified it is saved to following path: "
                                             "<output_directory>/<class_name>/<video_file_name>_<frame_number>.jpg")
    ap.add_argument("-i", "--video_file_path", required=True,
                    help="Path for video file")
    ap.add_argument("-o", "--output_directory", required=False,
                    help="Path for directory where to output images")
    return ap.parse_args()


def get_class_specific_directory_path(class_name, output_directory_path):
    return path.join(output_directory_path, class_name)


def save_frame(frame, output_directory, video_file_name, frame_number):
    if not path.exists(output_directory):
        os.mkdir(output_directory)

    file_path = path.join(output_directory, "{}_{}.jpg".format(video_file_name, str(frame_number)))
    cv2.imwrite(file_path, frame)


def classify_and_save_frame(frame, class_name, video_file_name, frame_num):
    output_directory = get_class_specific_directory_path(class_name, args.output_directory)
    save_frame(frame, output_directory, video_file_name, frame_num)


def put_key_mapping_text(frame):
    cv2.putText(frame, 'a - not_correct', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'd - correct', (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


def get_frame_class(key):
    if not 0 <= key <= 255:
        return None

    return key_mapping.get(chr(key))


def main(args):
    if not path.exists(args.video_file_path):
        print("File () doesn't exist".format(args.video_file_path))

    frame_num = 1
    cap = cv2.VideoCapture(args.video_file_path)
    video_file_name = path.splitext(path.basename(args.video_file_path))[0]

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret < 0:
            break

        # Display the resulting frame
        put_key_mapping_text(frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKeyEx(10)

        frame_class = get_frame_class(key)
        if frame_class is not None:
            classify_and_save_frame(frame, frame_class, video_file_name, frame_num)

        frame_num += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)
