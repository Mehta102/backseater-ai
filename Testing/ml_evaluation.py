# ML evaluation - generate metrics and plots
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# all 8 ways to win
WIN_LINES = [
    [0,1,2], [3,4,5], [6,7,8],
    [0,3,6], [1,4,7], [2,5,8],
    [0,4,8], [2,4,6]
]

def generate_data():
    # create training scenarios
    X, y = [], []
    
    for _ in range(300):
        board = [' '] * 9
        
        # simulate random moves
        for i in range(random.randint(1, 7)):
            empty = [j for j, cell in enumerate(board) if cell == ' ']
            if empty:
                board[random.choice(empty)] = 'X' if i % 2 == 0 else 'O'
        
        empty = [j for j, cell in enumerate(board) if cell == ' ']
        if not empty:
            continue
        
        pos = random.choice(empty)
        
        # count threats for both players
        my_threats = sum(1 for line in WIN_LINES 
                        if [board[i] for i in line].count('X') == 2 
                        and [board[i] for i in line].count(' ') == 1)
        opp_threats = sum(1 for line in WIN_LINES 
                         if [board[i] for i in line].count('O') == 2 
                         and [board[i] for i in line].count(' ') == 1)
        
        # extract features
        corners = [0, 2, 6, 8]
        features = [
            board.count('X'),
            board.count('O'),
            board.count(' '),
            1 if board[4] == 'X' else 0,
            sum(1 for i in corners if board[i] == 'X'),
            my_threats,
            opp_threats,
            1 if pos == 4 else 0,
            1 if pos in corners else 0
        ]
        
        # label based on game state
        if my_threats >= 2:
            label = 'fork'
        elif my_threats == 1:
            label = 'win'
        elif opp_threats >= 2:
            label = 'danger'
        elif opp_threats == 1:
            label = 'block'
        elif pos == 4:
            label = 'center'
        elif pos in corners:
            label = 'corner'
        else:
            label = 'neutral'
        
        X.append(features)
        y.append(label)
    
    return X, y

# generate data and train model
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=8, random_state=42)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.1%}")

# save report
report = classification_report(y_test, y_pred)
with open('ml_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.1%}\n\n")
    f.write(report)

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# per-class accuracy
class_acc = {}
for cls in model.classes_:
    mask = np.array(y_test) == cls
    if mask.sum() > 0:
        correct = sum((np.array(y_test)[mask] == np.array(y_pred)[mask]))
        class_acc[cls] = (correct / mask.sum()) * 100

plt.barh(list(class_acc.keys()), list(class_acc.values()), color='#58a6ff')
plt.xlabel('Accuracy (%)')
plt.title('Per-Class Accuracy')
plt.tight_layout()
plt.savefig('class_accuracy.png', dpi=300)
plt.close()

print("Saved: confusion_matrix.png, class_accuracy.png, ml_report.txt")
