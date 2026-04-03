# timing test - measure pipeline performance
import time
import json
import csv
from backseater import grab_region, read_board, AICommentary

# load ROI from saved file
with open('roi.json', 'r') as f:
    roi = json.load(f)
x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']

ai = AICommentary()

# store timing results
capture_times = []
detect_times = []
predict_times = []
total_times = []

# run 30 iterations
for i in range(30):
    start = time.time()
    
    # capture screen
    t0 = time.time()
    frame = grab_region(x, y, w, h)
    capture_times.append((time.time() - t0) * 1000)
    
    # detect board
    t1 = time.time()
    board = read_board(frame)
    detect_times.append((time.time() - t1) * 1000)
    
    # predict with AI
    t2 = time.time()
    if board.count(' ') < 9:
        ai.predict(board, 'X', 4)
    predict_times.append((time.time() - t2) * 1000)
    
    total_times.append((time.time() - start) * 1000)

# print averages
print(f"Screen Capture: {sum(capture_times)/30:.1f}ms avg")
print(f"Board Detection: {sum(detect_times)/30:.1f}ms avg")
print(f"ML Prediction: {sum(predict_times)/30:.1f}ms avg")
print(f"Total: {sum(total_times)/30:.1f}ms avg")

# save to csv
with open('timing_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'capture_ms', 'detection_ms', 'prediction_ms', 'total_ms'])
    for i in range(30):
        writer.writerow([i+1, capture_times[i], detect_times[i], predict_times[i], total_times[i]])

print("Saved to timing_results.csv")
