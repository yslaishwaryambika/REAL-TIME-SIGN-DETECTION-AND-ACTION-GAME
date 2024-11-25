
import cv2
import numpy as np
import random
import time
import mediapipe as mp

class ASLGame:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Game states
        self.current_state = 'MENU'
        self.game_mode = None
        
        # Game variables
        self.letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.numbers = '0123456789'
        self.current_target = None
        self.score = 0
        self.streak = 0
        self.feedback_text = ""
        self.feedback_time = 0
        self.feedback_duration = 2
        self.show_guide = False
        self.guide_duration = 3
        self.guide_start_time = None
        
        # Colors
        self.colors = {
            'pink': (203, 192, 255),
            'blue': (255, 191, 0),
            'green': (156, 255, 156),
            'purple': (255, 156, 255),
            'yellow': (0, 255, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (0, 0, 255)
        }
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Timer settings
        self.timer_duration = 10
        self.start_time = None
        
        # Mouse handling
        self.mouse_x = 0
        self.mouse_y = 0
        self.clicked = False
        
        # Create window and set mouse callback
        cv2.namedWindow('ASL Learning Game')
        cv2.setMouseCallback('ASL Learning Game', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_x = x
        self.mouse_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True

    def handle_menu_interaction(self, letters_button, numbers_button):
        """Handle menu interactions with both mouse and keyboard"""
        key = cv2.waitKey(1) & 0xFF
        
        # Handle keyboard input
        if key == ord('l'):
            self.game_mode = 'letters'
            self.current_state = 'GAME'
            self.new_target()
            return
        elif key == ord('n'):
            self.game_mode = 'numbers'
            self.current_state = 'GAME'
            self.new_target()
            return
            
        # Handle mouse clicks
        if self.clicked:
            x, y = self.mouse_x, self.mouse_y
            
            # Check letters button
            if (letters_button[0][0] < x < letters_button[1][0] and 
                letters_button[0][1] < y < letters_button[1][1]):
                self.game_mode = 'letters'
                self.current_state = 'GAME'
                self.new_target()
            
            # Check numbers button
            elif (numbers_button[0][0] < x < numbers_button[1][0] and 
                  numbers_button[0][1] < y < numbers_button[1][1]):
                self.game_mode = 'numbers'
                self.current_state = 'GAME'
                self.new_target()
            
            self.clicked = False

    def new_target(self):
        """Generate new target and reset timer"""
        if self.game_mode == 'letters':
            self.current_target = random.choice(self.letters)
        else:
            self.current_target = random.choice(self.numbers)
        self.start_time = time.time()
        self.show_guide = False

    def show_feedback(self, is_correct):
        """Show feedback and handle guide display"""
        self.feedback_text = "Good job!" if is_correct else "Uh-oh! Try again"
        self.feedback_time = time.time()
        if not is_correct:
            self.show_guide = True
            self.guide_start_time = time.time()

    def draw_hand_guide(self, frame):
        """Draw guide for correct hand position"""
        h, w = frame.shape[:2]
        
        # Draw guide box
        guide_box = (w-300, 100, 250, 300)
        cv2.rectangle(frame,
                     (guide_box[0], guide_box[1]),
                     (guide_box[0] + guide_box[2], guide_box[1] + guide_box[3]),
                     self.colors['blue'],
                     2)
        
        # Draw guide text
        guide_text = f"Correct position for '{self.current_target}'"
        cv2.putText(frame, guide_text,
                   (guide_box[0], guide_box[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   self.colors['black'], 2)
        
        # Draw reference points (simplified example)
        center_x = guide_box[0] + guide_box[2] // 2
        center_y = guide_box[1] + guide_box[3] // 2
        
        # Draw main points
        points = [
            (center_x, center_y),  # Palm center
            (center_x, center_y - 50),  # Middle finger
            (center_x - 30, center_y - 40),  # Index finger
            (center_x + 30, center_y - 40),  # Ring finger
            (center_x - 45, center_y - 20),  # Thumb
            (center_x + 45, center_y - 30),  # Pinky
        ]
        
        # Draw points and connections
        for point in points:
            cv2.circle(frame, point, 5, self.colors['yellow'], -1)
            cv2.circle(frame, point, 7, self.colors['black'], 1)
        
        # Draw connections
        for i in range(1, len(points)):
            cv2.line(frame, points[0], points[i],
                    self.colors['purple'], 2)

    def draw_menu(self, frame):
        """Draw menu interface"""
        h, w = frame.shape[:2]
        
        # Draw background
        cv2.rectangle(frame, (0, 0), (w, h), self.colors['blue'], -1)
        
        # Draw title
        title = "ASL Learning Game"
        cv2.putText(frame, title, (w//2-200, h//3),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors['white'], 3)
        
        # Draw buttons
        button_height = 80
        button_width = 300
        
        # Letters button
        letters_button = ((w//2-button_width-20, h//2),
                         (w//2-20, h//2+button_height))
        cv2.rectangle(frame, letters_button[0], letters_button[1],
                     self.colors['purple'], -1)
        cv2.putText(frame, "Learn Letters (L)",
                   (letters_button[0][0]+50, letters_button[0][1]+50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['white'], 2)
        
        # Numbers button
        numbers_button = ((w//2+20, h//2),
                         (w//2+button_width+20, h//2+button_height))
        cv2.rectangle(frame, numbers_button[0], numbers_button[1],
                     self.colors['green'], -1)
        cv2.putText(frame, "Learn Numbers (N)",
                   (numbers_button[0][0]+50, numbers_button[0][1]+50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['white'], 2)
        
        # Draw instructions
        instructions = "Click or Press 'L' for Letters | 'N' for Numbers | 'Q' to Quit"
        cv2.putText(frame, instructions,
                   (w//2-400, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['white'], 2)
        
        return letters_button, numbers_button

    def draw_game_interface(self, frame):
        """Draw game interface"""
        h, w = frame.shape[:2]
        
        # Draw header
        cv2.rectangle(frame, (0, 0), (w, 100), self.colors['pink'], -1)
        
        # Draw score and streak
        cv2.putText(frame, f"Score: {self.score}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['black'], 2)
        cv2.putText(frame, f"Streak: {self.streak}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['black'], 2)
        
        # Draw current target
        target_text = f"Show me: {self.current_target}"
        cv2.rectangle(frame, (w-300, 20), (w-20, 80),
                     self.colors['purple'], -1)
        cv2.putText(frame, target_text, (w-280, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['white'], 2)
        
        # Draw feedback if active
        if time.time() - self.feedback_time < self.feedback_duration:
            cv2.putText(frame, self.feedback_text,
                       (w//2-100, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                       self.colors['green'] if "Good" in self.feedback_text else self.colors['red'],
                       3)
        
        # Draw guide if active
        if self.show_guide:
            if time.time() - self.guide_start_time < self.guide_duration:
                self.draw_hand_guide(frame)
            else:
                self.show_guide = False
        
        # Draw timer bar
        if self.start_time:
            elapsed = time.time() - self.start_time
            remaining = max(0, self.timer_duration - elapsed)
            progress = remaining / self.timer_duration
            
            bar_width = int(w * 0.8)
            bar_height = 20
            bar_x = (w - bar_width) // 2
            bar_y = h - 50
            
            # Background bar
            cv2.rectangle(frame,
                         (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         self.colors['black'],
                         -1)
            
            # Progress bar
            progress_width = int(bar_width * progress)
            cv2.rectangle(frame,
                         (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + bar_height),
                         self.colors['green'] if progress > 0.3 else self.colors['red'],
                         -1)

    def process_keyboard_input(self, key):
        """Process keyboard input during gameplay"""
        if key == -1:
            return True
            
        try:
            key_char = chr(key).upper()
            
            if key_char == self.current_target:
                self.score += 10
                self.streak += 1
                self.show_feedback(True)
                self.new_target()
            elif key_char in (self.letters if self.game_mode == 'letters' else self.numbers):
                self.streak = 0
                self.show_feedback(False)
        except ValueError:
            pass
        
        return True

    def run(self):
        """Main game loop"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            if self.current_state == 'MENU':
                letters_button, numbers_button = self.draw_menu(frame)
                self.handle_menu_interaction(letters_button, numbers_button)
            
            elif self.current_state == 'GAME':
                # Process hands
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=self.colors['purple'][::-1], thickness=2, circle_radius=4),
                            self.mp_drawing.DrawingSpec(color=self.colors['blue'][::-1], thickness=2, circle_radius=2)
                        )
                
                # Draw game interface
                self.draw_game_interface(frame)
                
                # Check timer
                if self.start_time and time.time() - self.start_time > self.timer_duration:
                    self.streak = 0
                    self.show_feedback(False)
                    self.new_target()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.process_keyboard_input(key):
                    break
            
            # Display frame
            cv2.imshow('ASL Learning Game', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    try:
        game = ASLGame()
        game.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

