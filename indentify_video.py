# เรียกใช้งาน Modules
import face_recognition
import cv2

# ระบุตำแหน่งของ Video ที่ต้องการนำมาตรวจสอบ
video_capture = cv2.VideoCapture("./video/test1.mp4")

# สร้างการเรียนรู้ใบหน้าโดยการดึงรูปภาพของใบหน้ามาจดจำ encoding
# สามารถสร้างได้ไม่จำกัด
person1_image = face_recognition.load_image_file("./img/known/tim.png")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("./img/known/Steve Jobs.jpg")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

# บันทึกใบหน้าลง Array เพื่อไว้ตรวจสอบ
known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding,
]

# ใส่ชื่อให้กับบุคคลที่ต้องการบันทึกโดยเรียนชื่อตาม encoding array ด้านบน
known_face_names = [
    "Tim Cook",
    "Steve Job"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# เป็น loop ตรวจสอบใบหน้าเหมือน indentify_picture แต่ปรับแต่งสำหรับ Video โดยที่การดึงทีละ Frame ของ Video ไปตรวจสอบ โดยไม่จำกัด loop
while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    
    # กดปุ่ม q เพื่อหยุดการทำงาน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()