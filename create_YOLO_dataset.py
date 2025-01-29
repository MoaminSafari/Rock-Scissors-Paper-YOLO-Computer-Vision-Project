import cv2
import os
from os import sep
import mediapipe as mp


def find_hand(RGBimg, results, margin=20):
    if results.multi_hand_landmarks:
        xList, yList = [], []
        myHand = results.multi_hand_landmarks[0]
        for lm in myHand.landmark:
            h, w, _ = RGBimg.shape
            px, py = int(lm.x * w), int(lm.y * h)
            xList.append(px)
            yList.append(py)
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax - xmin, ymax - ymin
        return xmin - margin, ymin - margin, boxW + 2 * margin, boxH + 2 * margin
    return None


class CamDataset:
    def __init__(self, base_path, image_dsize, class_id, image_format='jpg'):
        self.class_id = class_id  # Index of class in YOLO format
        self.image_dsize = image_dsize
        self.image_format = image_format

        # Paths following YOLO dataset structure
        self.images_train_path = os.path.join(base_path, 'images', 'train')
        self.images_val_path = os.path.join(base_path, 'images', 'val')
        self.labels_train_path = os.path.join(base_path, 'labels', 'train')
        self.labels_val_path = os.path.join(base_path, 'labels', 'val')
        self.images_bb_path = os.path.join(base_path, 'images', 'bb')

        # Ensure directories exist
        for path in [self.images_train_path, self.images_val_path, self.labels_train_path, self.labels_val_path, self.images_bb_path]:
            os.makedirs(path, exist_ok=True)

    def start_streaming(self, label, name, mode='train'):
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(static_image_mode=False, max_num_hands=1,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5)

        img_counter = 0
        cam = cv2.VideoCapture(0)

        print("Press 'q' to quit, 's' to save data")
        while True:
            _, img = cam.read()
            img = cv2.resize(img, self.image_dsize)
            RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(RGBimg)
            hand = find_hand(RGBimg, results)

            if hand:
                x, y, w, h = hand
                main_img = img.copy()
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if cv2.waitKey(1) & 0xFF == ord("s"):
                    # Select train or val directory
                    img_folder = self.images_train_path if mode == 'train' else self.images_val_path
                    label_folder = self.labels_train_path if mode == 'train' else self.labels_val_path
                    bb_folder = self.images_bb_path

                    image_path = os.path.join(img_folder,   f"{mode}_{label}_{img_counter}_{name}.{self.image_format}")
                    label_path = os.path.join(label_folder, f"{mode}_{label}_{img_counter}_{name}.txt")
                    bb_path = os.path.join(bb_folder,    f"bb_{mode}_{label}_{img_counter}_{name}.{self.image_format}")

                    # YOLO format: class_id x_center y_center width height (normalized)
                    x_center = (x + w / 2) / self.image_dsize[0]
                    y_center = (y + h / 2) / self.image_dsize[1]
                    width = w / self.image_dsize[0]
                    height = h / self.image_dsize[1]

                    annotation_data = f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                    cv2.imwrite(image_path, main_img)
                    with open(label_path, "w") as annotation_file:
                        annotation_file.write(annotation_data)
                    cv2.imwrite(bb_path, img)

                    print(f"Saved image: {image_path}, annotation: {label_path}")
                    img_counter += 1

            elif cv2.waitKey(1) & 0xFF == ord("s"):
                print("No objects detected")

            cv2.imshow("Webcam Stream", cv2.resize(img, (800, 600)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    BASE_PATH = os.path.join('.', 'dataset')
    IMAGE_SIZE = (640, 640)
    CLASS_ID = 2        # Change based on class index in data.yaml. paper = 0 , rock = 1, scissors = 2
    LABEL = 'scissors'     # paper, rock, scissors
    NAME = 'amin'        # ali, amin, maryam
    MODE = 'val'      # train, val

    cam = CamDataset(BASE_PATH, IMAGE_SIZE, CLASS_ID)
    cam.start_streaming(label=LABEL, name=NAME, mode=MODE)  # Change to 'val' for validation data
