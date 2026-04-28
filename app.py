import pygame
import numpy as np
import cv2
import joblib

model = joblib.load("knn_model.pkl")

pygame.init()

FONT = pygame.font.SysFont("Arial", 60)

WIDTH = 640
HEIGHT = 480

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("KNN Digit Recognition")

screen.fill((0,0,0))

drawing = False
last_pos = None

# Digit labels
LABELS = {
0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",
5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"
}

def draw(screen, start, end, width=18):
    pygame.draw.line(screen, (255,255,255), start, end, width)


def get_image():

    data = pygame.surfarray.array3d(screen)

    data = np.rot90(data)
    data = np.flipud(data)

    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    data = 255 - data

    _, thresh = cv2.threshold(data, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    digit = thresh[y:y+h, x:x+w]

    digit = cv2.resize(digit, (28,28))

    digit = digit.reshape(1,784)

    return digit


def predict_digit():

    img = get_image()

    if img is None:
        return

    prediction = model.predict(img)[0]

    word = LABELS[prediction]

    x, y = pygame.mouse.get_pos()

    predictions.append((word, x, y))


running = True
predictions = []

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            predict_digit()

        elif event.type == pygame.MOUSEMOTION and drawing:
            mouse_pos = pygame.mouse.get_pos()
            draw(screen, last_pos, mouse_pos)
            last_pos = mouse_pos

        elif event.type == pygame.KEYDOWN:

            # press C to clear
            if event.key == pygame.K_c:
                screen.fill((0,0,0))
                predictions.clear()

    # display predictions
    for word, x, y in predictions:
        text_surface = FONT.render(word, True, (255,0,0))
        screen.blit(text_surface, (x, y))

    pygame.display.update()

pygame.quit()