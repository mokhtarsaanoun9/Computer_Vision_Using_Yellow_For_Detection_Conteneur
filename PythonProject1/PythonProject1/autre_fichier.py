import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import shutil
import matplotlib.pyplot as plt


model = YOLO(r"C:\Users\mokhtar\PythonProject1\runs\detect\train2\weights\best.pt")


shutil.copy(r"C:\Users\mokhtar\PythonProject1\runs\detect\train2\weights\best.pt",
            r"C:\Users\mokhtar\PythonProject1\model\mon_model.pt")
print("Modèle enregistré avec succès !")


image_path = r"C:\Users\mokhtar\PythonProject1\detection\test\images\700135891807221341_0050_jpg.rf.0d331182340c4db39c468e6b03035b6c.jpg"


results = model(image_path)

result_img = results[0].plot()


plt.figure(figsize=(10, 10))
plt.imshow(result_img)
plt.axis("off")
plt.title("Image avec détections")
plt.show()
