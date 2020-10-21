# เรียกใช้งาน Modules
import face_recognition
import cv2
from PIL import Image, ImageDraw

# สร้างการเรียนรู้ใบหน้าโดยการดึงรูปภาพของใบหน้ามาจดจำ encoding
# สามารถสร้างได้ไม่จำกัด
image_of_tim = face_recognition.load_image_file('./img/known/Tim Cook.png')
tim_face_encoding = face_recognition.face_encodings(image_of_tim)[0]

image_of_steve = face_recognition.load_image_file('./img/known/Steve Jobs.jpg')
steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]

# บันทึกใบหน้าลง Array เพื่อไว้ตรวจสอบ
known_face_encodings = [
  tim_face_encoding,
  steve_face_encoding,
  nat_face_encoding,
]

# ใส่ชื่อให้กับบุคคลที่ต้องการบันทึกโดยเรียนชื่อตาม encoding array ด้านบน
known_face_names = [
  "Tim Cook",
  "Steve Jobs",
]

# ระบุตำแหน่งภาพที่ต้องการนำมาเปรียบเทียบ
compare_image = face_recognition.load_image_file('img/people/people1.png')

# เป็นการนำภาพที่ต้องการเปรียบเทียบมา encoding
face_locations = face_recognition.face_locations(compare_image)
face_encodings = face_recognition.face_encodings(compare_image, face_locations)

# เป็นการเตรียมสร้างกรอบครอบรูปภาพเพื่อบอกชื่อและวาดกรอบตรงใบหน้า
pil_image = Image.fromarray(compare_image)
draw = ImageDraw.Draw(pil_image)

# เป็น loop ตรวจสอบใบหน้าและตีกรอบหน้าใบหน้าพร้อมขึ้นชื่อบุคคล ถ้ามีในระบบจะขึ้นชื่ออกมา ถ้าไม่มีจะเป็น Unknown แทน 
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
  name = "Unknown"
  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# บันทึกรูปที่เปรียบเทียบและตรวจสอบเรียบร้อยเป็นไฟล์ jpg
pil_image.save('identify.jpg')

