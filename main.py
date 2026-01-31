import cv2, os, json, numpy as np, shutil
from pathlib import Path
from tqdm import tqdm

# === Directory Setup ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "face_model.yml"
LABELS_FILE = MODEL_DIR / "labels.json"
PROTO_FILE = MODEL_DIR / "deploy.prototxt"
CAFFE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# ensure folders exist
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# === Load DNN Model ===
def load_detection_model():
    """Load Caffe DNN model for face detection."""
    if not (PROTO_FILE.exists() and CAFFE_MODEL.exists()):
        raise FileNotFoundError(
            "Model files missing: deploy.prototxt or res10_300x300.caffemodel"
        )

    net = cv2.dnn.readNetFromCaffe(str(PROTO_FILE), str(CAFFE_MODEL))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


# === Detect faces ===
def detect_faces(net, frame, thr=0.6):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    result = net.forward()

    output_boxes = []
    for i in range(result.shape[2]):
        conf = result[0, 0, i, 2]
        if conf > thr:
            box = result[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            output_boxes.append((x1, y1, x2 - x1, y2 - y1))

    return output_boxes


# === Preprocess Face ===
def prepare_face(gray, box):
    x, y, w, h = box
    crop = gray[y:y + h, x:x + w]
    crop = cv2.resize(crop, (150, 150))
    crop = cv2.equalizeHist(crop)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    return crop


# === Collect training data ===
def record_faces(user_name, net):
    user_folder = DATA_DIR / user_name
    user_folder.mkdir(exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Webcam not found.")

    print(f"[INFO] Recording images for '{user_name}'. Press 'q' to exit.")
    img_count, LIMIT = 0, 200

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(net, frame)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face = prepare_face(gray, (x, y, w, h))
            cv2.imwrite(str(user_folder / f"img_{img_count:04d}.jpg"), face)
            img_count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{img_count}/{LIMIT}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Collecting Faces", frame)
        if cv2.waitKey(80) & 0xFF == ord('q') or img_count >= LIMIT:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[✓] Collected {img_count} images.")


# === Train LBPH Model ===
def build_model():
    pics, lbls = [], []
    name_to_id = {}
    current_id = 0

    folders = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    total = sum(len(list(d.glob("*.jpg"))) for d in folders)

    if total == 0:
        print("No training data available.")
        return

    with tqdm(total=total, desc="Training", ncols=70) as bar:
        for folder in folders:
            uname = folder.name
            images = list(folder.glob("*.jpg"))
            if not images:
                continue

            name_to_id[uname] = current_id

            for img_path in images:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    bar.update(1)
                    continue

                img = cv2.resize(img, (150, 150))
                img = cv2.equalizeHist(img)

                pics.append(img)
                lbls.append(current_id)
                bar.update(1)

            current_id += 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(pics, np.array(lbls))
    recognizer.write(str(MODEL_FILE))

    with open(LABELS_FILE, "w") as f:
        json.dump(name_to_id, f)

    print("[✓] Model trained.")
    print("Trained users:", ", ".join(name_to_id.keys()))


# === Delete User ===
def remove_user():
    users = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if not users:
        print("No users to delete.")
        return

    print("\n=== Select user to delete ===")
    for idx, user in enumerate(users, 1):
        print(f"{idx}. {user.name}")

    try:
        opt = int(input("Choice: "))
        if opt < 1 or opt > len(users):
            print("Invalid input.")
            return
    except:
        print("Invalid number.")
        return

    folder = users[opt - 1]
    uname = folder.name

    shutil.rmtree(folder)
    print(f"[✓] Deleted user: {uname}")

    # update labels file
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            labels = json.load(f)

        labels.pop(uname, None)
        new_map = {name: idx for idx, name in enumerate(labels.keys())}

        with open(LABELS_FILE, "w") as f:
            json.dump(new_map, f, indent=4)

        print("[✓] Updated label map.")


# === Authenticate ===
def verify_user(net, thr=80):
    if not (MODEL_FILE.exists() and LABELS_FILE.exists()):
        print("Train model first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))

    with open(LABELS_FILE) as f:
        label_map = json.load(f)

    rev_map = {v: k for k, v in label_map.items()}
    print("[INFO] Loaded:", rev_map)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Webcam not found.")

    stable, HIT_REQ = 0, 8
    final_user = None

    print("Press 'q' to quit.")

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(net, frame)

        for (x, y, w, h) in faces:
            face = prepare_face(gray, (x, y, w, h))
            lid, conf = recognizer.predict(face)
            uname = rev_map.get(lid, "Unknown")

            if conf <= thr:
                stable += 1
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
                stable = 0

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{uname} ({conf:.1f})", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if stable >= HIT_REQ:
                final_user = uname

        if final_user:
            cv2.putText(frame, f"AUTHENTICATED: {final_user}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 200, 0), 3)
            cv2.imshow("Verify", frame)
            cv2.waitKey(1200)
            break

        cv2.imshow("Verify", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if not final_user:
        print("[×] Access Denied")
    else:
        print(f"[✓] Access Granted: {final_user}")


# === Main Menu ===
def main():
    print("[INFO] Loading face detector...")
    net = load_detection_model()
    print("[✓] DNN Model Loaded!")

    while True:
        print("\n=== FACE SYSTEM ===")
        print("1. Register User")
        print("2. Train Model")
        print("3. Authenticate")
        print("4. Delete User")
        print("5. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            uname = input("Enter new username: ").strip()
            folder = DATA_DIR / uname

            if folder.exists():
                print("User already exists.")
            else:
                record_faces(uname, net)

        elif choice == "2":
            build_model()

        elif choice == "3":
            verify_user(net)

        elif choice == "4":
            remove_user()

        elif choice == "5":
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
