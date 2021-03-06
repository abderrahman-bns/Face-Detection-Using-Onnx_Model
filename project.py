import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
#Code to get the right boxes is shown below:
def area_of(left_top,right_bottom):
    #compute the area of rectangle given two corners <<numpy.clip (a, a_min, a_max, out = None)>>
    hw = np.clip(right_bottom-left_top,0.0,None) #La fonction est utilisée pour découper (limiter) les valeurs dans un tableau.
    #Par exemple, si un intervalle de [0, 1] est spécifié, les valeurs inférieures à 0 deviennent 0 et les valeurs supérieures à 1 deviennent 1.
    return hw[...,0]*hw[...,1]
#Return intersection-over-union (Jaccard index) of boxes.
'''
l'IOU est une métrique importante pour décider de la prédiction d'objet des modèles d'apprentissage en profondeur.
'''
def iou_of(boxes0,boxes1,eps=1e-5):
    '''
   np.maximum() : Est utilisée pour trouver le maximum par élément des éléments du tableau.
Il compare deux tableaux et renvoie un nouveau tableau contenant les maxima par élémen
    '''
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

#Perform hard non-maximum-supression to filter out boxes with iou greater than threshold
'''
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
'''
#General Threshold for the IOU can be 0.5.  Normally IOU>0.5 is considered a good prediction
def hard_nms(box_scores,iou_threshold,top_k=-1,candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    #Effectuez un tri indirect le long de l'axe
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # An Intersection over Union score > 0.5 is normally considered a “good” prediction.
        current = indexes[-1]
        #ajoute un élément à la fin de la liste.
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        ##Return intersection-over-union (Jaccard index) of boxes.
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        #Normally IOU>0.5 is considered a good prediction
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

#Select boxes that contain human faces
'''
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
'''
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    #The shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
    #shape is a tuple that gives dimensions of the array.
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        #joignez une séquence de tableaux le long d'un axe existant.
        #numpy.reshape(ligne,colonne):Donne une nouvelle forme à un tableau sans modifier ses données.
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        # Perform hard non-maximum-supression to filter out boxes with iou greater than threshold
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        #L'extend () étend la liste en ajoutant tous les éléments de la liste (passés en argument) à une fin.
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    '''
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    '''
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]





'''
Les images sont simplement des tableaux numpy 
, qui prennent en charge une variété de types de données 1 
, à savoir les «dtypes». Pour éviter de fausser les intensités d'image (voir Redimensionnement des valeurs d'intensité )
'''
video_capture = cv2.VideoCapture(0)
#prepare the ONNX model and create an ONNX inference session.
onnx_model = onnx.load('ultra_light_640.onnx')

predictor = prepare(onnx_model)

ort_session = ort.InferenceSession('ultra_light_640.onnx')

input_name = ort_session.get_inputs()[0].name


while True:
    ret, frame = video_capture.read()
    # size photo
    h, w, _ = frame.shape
    # conversion d'espace colorimétrique.
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # Redimmensionner une image
    img = cv2.resize(img,(640,480))

    img_mean = np.array([127,127,127])
    # Normalize image:  La normalisation de l'image  est un processus dans lequel nous modifions la plage des valeurs d'intensité des pixels pour rendre l'image plus familière ou normale aux sens
    img = (img - img_mean)/128
    # We can use the transpose() function to get the transpose of an array.
    img = np.transpose(img, [2,0,1])
    #Développez la forme d'un tableau.
    img = np.expand_dims(img,axis=0) #

    img = img.astype(np.float32) #
    #confidences :contient une liste de niveau de confiance pour chaque case à l'intérieur de la boxes variable
    #boxes : valeur contient toutes les boîtes générées, nous devrons identifier les boîtes avec une forte probabilité
    # de contenir un visage et supprimer les doublons
    confidences, boxes = ort_session.run(None, {input_name: img})

    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"face: {labels[i]}"
        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

