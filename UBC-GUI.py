import sys, os, joblib, numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
import pandas as pd

TEAL, GREEN, LIGHT = "#326B7E", "#20C997", "#7DC0C6"

# Load saved models
models = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "UBC_models.pkl"))
XGB_model, AdaB_model = models["XGBoost"], models["AdaBoost"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bearing Capacity Prediction")
        self.resize(700, 550)

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)

        # Header
        title = QLabel("Shallow Foundations UBC - Cohesionless Soil")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"background:{TEAL};color:white;font-size:22px;font-weight:bold;padding:15px")
        main.addWidget(title)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self.prediction_tab(), "Prediction")
        tabs.addTab(QLabel("This application was developed as a part of the Pre-Master Year project.",
                           alignment=Qt.AlignCenter), "About")
        main.addWidget(tabs)

    def prediction_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Inputs + diagram
        top = QHBoxLayout()
        inputs = QGroupBox("Foundation && Soil Parameters")
        grid = QGridLayout(inputs)

        self.B, self.D, self.LB, self.gamma, self.phi = [QDoubleSpinBox() for _ in range(5)]

        for w, r, value, suffix, maximum in [
            (self.B, 0, 1.5, " m", 100),
            (self.D, 1, 1, " m", 100),
            (self.LB, 2, 2, "", 20),
            (self.gamma, 3, 18, " kN/m³", 30),
            (self.phi, 4, 32, " °", 50)
        ]:
            w.setRange(0, maximum)
            w.setValue(value)
            w.setDecimals(3)
            w.setSuffix(suffix)

        for row, (name, widget) in enumerate([
            ("Width B", self.B), ("Depth D", self.D), ("L/B", self.LB),
            ("Unit Weight γ", self.gamma), ("Friction Angle φ", self.phi)
        ]):
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(widget, row, 1)

        top.addWidget(inputs)

        diagram = QGroupBox("Foundation")
        d = QLabel("       L\n  ┌──────────┐\n  │ FOUNDATION │\n  └──────────┘\n       ↓ D\n───────────────\n      SOIL")
        d.setAlignment(Qt.AlignCenter)
        d.setStyleSheet(f"color:{TEAL};font-size:16px;background:#f7fbfc;padding:20px")
        QVBoxLayout(diagram).addWidget(d)
        top.addWidget(diagram)

        layout.addLayout(top)

        # Predict button
        button = QPushButton("PREDICT BEARING CAPACITY")
        button.setStyleSheet(f"background:{GREEN};color:white;font-weight:bold;padding:12px")
        button.clicked.connect(self.predict)
        layout.addWidget(button)

        # Results
        results = QGroupBox("Prediction Results")
        r = QHBoxLayout(results)

        self.xgb, self.ada, self.sr = [QLabel(f"{x}\n--- kPa", alignment=Qt.AlignCenter)
                                       for x in ["XGBoost", "AdaBoost", "Symbolic Regression"]]

        for label in (self.xgb, self.ada, self.sr):
            label.setStyleSheet(f"color:{TEAL};border:1px solid {LIGHT};padding:15px;font-weight:bold")
            r.addWidget(label)

        layout.addWidget(results)
        return tab

    def predict(self):
        B, D, LB, g, phi = self.B.value(), self.D.value(), self.LB.value(), self.gamma.value(), self.phi.value()

        # print inputs
        print('B = ', B)
        print('D = ', D)
        print('LB = ', LB)
        print('g = ', g)
        print('phi = ', phi)

        # ML models
        X = pd.DataFrame([[B, D, g, phi, LB]], columns=['B', 'D', 'g', 'phi', 'LB'])
        xgb = XGB_model.predict(X)[0]
        ada = AdaB_model.predict(X)[0]

        # SR equation
        p = np.radians(phi)
        Nq = np.exp(np.pi*np.tan(p)) * np.tan(np.pi/4 + p/2)**2
        Ngamma = 2*(Nq + 1)*np.tan(p)
        gammaD = g * D
        sr = (Ngamma + gammaD**2*(D/B - np.sin(p)**2)) * (B*5.938716 + gammaD)

        self.xgb.setText(f"XGBoost\n{xgb:.2f} kPa")
        self.ada.setText(f"AdaBoost\n{ada:.2f} kPa")
        self.sr.setText(f"Symbolic Regression\n{sr:.2f} kPa")


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())