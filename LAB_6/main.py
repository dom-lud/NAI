"""
Projekt: Detekcja gestów dłoni - sterowanie „odtwarzaczem” gestami
Autorzy:
- Dominik Ludwiński
- Bartosz Dembowski

Opis:
Program działa na żywo na strumieniu wideo z kamery (OpenCV) i rozpoznaje gesty dłoni
na podstawie landmarków z MediaPipe HandLandmarker (Tasks API). Na ekranie wyświetla
bieżący gest oraz stan „odtwarzacza” (PLAY/PAUSE, track, volume). Dodatkowo pokazuje
duży napis na środku po wykonaniu akcji.

Mapowanie gestów:
1) OPEN_PALM              -> PLAY
2) FIST                   -> PAUSE
3) PEACE (V)              -> NEXT (track + 1)
4) THREE (3 palce)        -> PREVIOUS (track - 1)
5) THUMB_UP lub PINCH     -> VOLUME UP (+5)
6) THUMB_DOWN             -> VOLUME DOWN (-5)

Sterowanie:
- ESC: wyjście

Wymagania:
pip install opencv-python mediapipe numpy
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model() -> None:
    """Pobiera plik modelu MediaPipe (.task), jeśli nie istnieje lokalnie."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe HandLandmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


@dataclass
class PlayerState:
    """Stan prostego „odtwarzacza” sterowanego gestami."""
    status: str = "STOP"
    track: int = 1
    volume: int = 50
    last_action: str = "-"
    overlay_text: str = ""
    overlay_until: float = 0.0


class GestureController:
    """
    Rozpoznaje gesty na podstawie landmarków dłoni i mapuje je na akcje.
    Stosuje:
    - stabilizację (gest musi utrzymać się N klatek),
    - cooldown (minimalny odstęp między wywołaniami akcji).
    """

    def __init__(self, cooldown_s: float = 0.8, stable_frames: int = 7) -> None:
        self.cooldown_s = cooldown_s
        self.stable_frames = stable_frames

        self._last_trigger_time = 0.0
        self._last_raw = "NONE"
        self._stable_count = 0

        self.state = PlayerState()

    def can_trigger(self) -> bool:
        """Czy można już wykonać kolejną akcję (cooldown)."""
        return (time.time() - self._last_trigger_time) >= self.cooldown_s

    def mark_triggered(self) -> None:
        """Zapisuje moment wykonania akcji."""
        self._last_trigger_time = time.time()

    @staticmethod
    def finger_is_up(lm: np.ndarray, tip: int, pip: int) -> bool:
        """Palec uznajemy za wyprostowany, gdy TIP jest wyraźnie wyżej niż PIP."""
        return lm[tip, 1] < (lm[pip, 1] - 0.02)

    @staticmethod
    def finger_is_down(lm: np.ndarray, tip: int, pip: int) -> bool:
        """Palec uznajemy za zgięty, gdy TIP jest wyraźnie niżej niż PIP."""
        return lm[tip, 1] > (lm[pip, 1] + 0.02)

    @staticmethod
    def thumb_is_up(lm: np.ndarray) -> bool:
        """Kciuk w górę: TIP powyżej IP + wyprost (dystans TIP–MCP)."""
        tip, ip, mcp = lm[4], lm[3], lm[2]
        up = tip[1] < (ip[1] - 0.02)
        dist = float(((tip[0] - mcp[0]) ** 2 + (tip[1] - mcp[1]) ** 2) ** 0.5)
        return up and (dist > 0.12)

    @staticmethod
    def thumb_is_down(lm: np.ndarray) -> bool:
        """Kciuk w dół: TIP poniżej IP + wyprost (dystans TIP–MCP)."""
        tip, ip, mcp = lm[4], lm[3], lm[2]
        down = tip[1] > (ip[1] + 0.02)
        dist = float(((tip[0] - mcp[0]) ** 2 + (tip[1] - mcp[1]) ** 2) ** 0.5)
        return down and (dist > 0.12)

    @staticmethod
    def hand_is_compact(lm: np.ndarray) -> bool:
        """
        Dłoń „zwarta” (pięść) – średnia odległość TIP-ów od nadgarstka mała.
        """
        wrist = lm[0]
        tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
        dists = [float(((p[0] - wrist[0]) ** 2 + (p[1] - wrist[1]) ** 2) ** 0.5) for p in tips]
        return float(np.mean(dists)) < 0.33

    @staticmethod
    def pinch(lm: np.ndarray) -> bool:
        """PINCH: kciuk bardzo blisko czubka wskazującego (backup dla VOLUME UP)."""
        thumb_tip = lm[4]
        index_tip = lm[8]
        dist = float(((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5)
        return dist < 0.06

    def recognize_raw(self, lm: np.ndarray) -> str:
        """Rozpoznanie pojedynczej klatki (bez stabilizacji)."""
        thumb_up = self.thumb_is_up(lm)
        thumb_down = self.thumb_is_down(lm)

        index_up = self.finger_is_up(lm, 8, 6)
        middle_up = self.finger_is_up(lm, 12, 10)
        ring_up = self.finger_is_up(lm, 16, 14)
        pinky_up = self.finger_is_up(lm, 20, 18)

        index_down = self.finger_is_down(lm, 8, 6)
        middle_down = self.finger_is_down(lm, 12, 10)
        ring_down = self.finger_is_down(lm, 16, 14)
        pinky_down = self.finger_is_down(lm, 20, 18)

        up4 = int(index_up) + int(middle_up) + int(ring_up) + int(pinky_up)

        if self.pinch(lm):
            return "THUMB_UP"

        if up4 == 4:
            return "OPEN_PALM"

        if index_up and middle_up and (not ring_up) and (not pinky_up) and (not thumb_up) and (not thumb_down):
            return "PEACE"

        if index_up and middle_up and ring_up and (not pinky_up) and (not thumb_up) and (not thumb_down):
            return "THREE"

        if thumb_up and (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
            return "THUMB_UP"

        if thumb_down and (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
            return "THUMB_DOWN"

        if index_down and middle_down and ring_down and pinky_down and (not thumb_up) and (not thumb_down):
            if self.hand_is_compact(lm):
                return "FIST"

        return "NONE"

    def recognize(self, lm: np.ndarray) -> str:
        """Stabilizacja: zwraca gest dopiero po N kolejnych klatkach tego samego wyniku."""
        raw = self.recognize_raw(lm)
        if raw == self._last_raw:
            self._stable_count += 1
        else:
            self._last_raw = raw
            self._stable_count = 1
        return raw if self._stable_count >= self.stable_frames else "NONE"

    def _set_overlay(self, text: str, duration_s: float = 1.2) -> None:
        """Ustawia overlay (duży napis na środku) na `duration_s` sekund."""
        self.state.overlay_text = text
        self.state.overlay_until = time.time() + duration_s

    def apply(self, gesture: str) -> None:
        """Wykonuje akcję zależnie od gestu."""
        if gesture == "OPEN_PALM":
            self.state.status = "PLAY"
            self.state.last_action = "PLAY"
            self._set_overlay("PLAY")

        elif gesture == "FIST":
            self.state.status = "PAUSE"
            self.state.last_action = "PAUSE"
            self._set_overlay("PAUSE")

        elif gesture == "PEACE":
            self.state.track += 1
            self.state.last_action = "NEXT"
            self._set_overlay("NEXT")

        elif gesture == "THREE":
            self.state.track = max(1, self.state.track - 1)
            self.state.last_action = "PREVIOUS"
            self._set_overlay("PREVIOUS")

        elif gesture == "THUMB_UP":
            self.state.volume = min(100, self.state.volume + 5)
            self.state.last_action = f"VOL {self.state.volume}"
            self._set_overlay(self.state.last_action)

        elif gesture == "THUMB_DOWN":
            self.state.volume = max(0, self.state.volume - 5)
            self.state.last_action = f"VOL {self.state.volume}"
            self._set_overlay(self.state.last_action)


def draw_hud(frame: np.ndarray, gesture: str, st: PlayerState) -> None:
    """Rysuje panel informacyjny w lewym górnym rogu oraz pasek głośności."""
    lines = [
        f"Gesture: {gesture}",
        f"Status : {st.status}",
        f"Track  : {st.track}",
        f"Volume : {st.volume}",
        f"Action : {st.last_action}",
        "ESC to exit",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30

    h, _ = frame.shape[:2]
    bar_x1, bar_y1 = 10, h - 30
    bar_x2, bar_y2 = 210, h - 10
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 255, 0), 2)
    fill_w = int((bar_x2 - bar_x1 - 4) * (st.volume / 100.0))
    cv2.rectangle(frame, (bar_x1 + 2, bar_y1 + 2), (bar_x1 + 2 + fill_w, bar_y2 - 2), (0, 255, 0), -1)


def draw_center_overlay(frame: np.ndarray, st: PlayerState) -> None:
    """Rysuje duży napis na środku ekranu (krótko po wykonaniu akcji)."""
    if not st.overlay_text or time.time() > st.overlay_until:
        return

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thick = 4

    (tw, th), _ = cv2.getTextSize(st.overlay_text, font, scale, thick)
    x = (w - tw) // 2
    y = (h + th) // 2

    cv2.putText(frame, st.overlay_text, (x, y), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, st.overlay_text, (x, y), font, scale, (0, 255, 0), thick, cv2.LINE_AA)


def main() -> None:
    """Uruchamia kamerę, detekcję dłoni i pętlę przetwarzania wideo."""
    ensure_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nie moge otworzyc kamery (VideoCapture(0)).")

    controller = GestureController(cooldown_s=0.8, stable_frames=7)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = landmarker.detect(mp_image)

        gesture = "NONE"

        if result.hand_landmarks:
            lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
            gesture = controller.recognize(lm)

            if gesture != "NONE" and controller.can_trigger():
                controller.apply(gesture)
                controller.mark_triggered()

            h, w = frame.shape[:2]
            for p in lm:
                cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 3, (0, 255, 0), -1)

        draw_hud(frame, gesture, controller.state)
        draw_center_overlay(frame, controller.state)

        cv2.imshow("Gestures Controller", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
