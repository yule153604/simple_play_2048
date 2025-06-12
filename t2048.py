import time
import random
import sys
import copy 
import math 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


URL = "https://2048.gg/zh"

ANIMATION_DELAY = 0.05 
WAIT_TIMEOUT = 10
AUTO_RESTART_GAME = True 

KEY_MAP = { Keys.ARROW_UP: "↑ UP", Keys.ARROW_DOWN: "↓ DOWN", Keys.ARROW_LEFT: "← LEFT", Keys.ARROW_RIGHT: "→ RIGHT", }
ALL_MOVES = [Keys.ARROW_DOWN, Keys.ARROW_LEFT, Keys.ARROW_RIGHT, Keys.ARROW_UP]
MOVE_PREFERENCE = { Keys.ARROW_DOWN: 0, Keys.ARROW_LEFT: 1, Keys.ARROW_RIGHT: 2, Keys.ARROW_UP: 3}



WEIGHT_MATRIX = [
    [ 2**4,  2**5,  2**6,  2**7 ],
    [ 2**3,  2**2,  2**1,  2**8 ], 
    [ 2**0,  2**1,  2**9, 2**10 ], 
    [ 2**15, 2**14, 2**12, 2**11 ] 
]
EMPTY_CELL_WEIGHT_FACTOR = 30000 
SMOOTHNESS_WEIGHT = 0.1
MONOTONICITY_WEIGHT = 1.0

CACHE = {} 





def transpose(matrix): return [list(row) for row in zip(*matrix)]
def reverse_rows(matrix): return [row[::-1] for row in matrix]
def compress(row):
     new_row = [i for i in row if i != 0]
     new_row.extend([0] * (4 - len(new_row)))
     return new_row
def merge(row):
     new_row = list(row)
     for i in range(3):
          if new_row[i] > 0 and new_row[i] == new_row[i+1]:
               new_row[i] *= 2
               new_row[i+1] = 0
     return new_row
def move_left(matrix):
    new_matrix = []
    for row in matrix: new_matrix.append(compress(merge(compress(row))))
    return new_matrix
def move_right(matrix): return reverse_rows(move_left(reverse_rows(matrix)))
def move_up(matrix): return transpose(move_left(transpose(matrix)))
def move_down(matrix): return transpose(move_right(transpose(matrix))) 

SIMULATE_MAP = { Keys.ARROW_LEFT: move_left, Keys.ARROW_RIGHT: move_right, Keys.ARROW_UP: move_up, Keys.ARROW_DOWN: move_down}

def simulate_move(matrix, move_key):
    if move_key not in SIMULATE_MAP: return matrix, False
    board_copy = copy.deepcopy(matrix) 
    new_matrix = SIMULATE_MAP[move_key](board_copy)
    return new_matrix, (new_matrix != matrix)

def get_empty_cells(matrix):
    """返回所有空格的 (row, col) 列表"""
    return [(r, c) for r in range(4) for c in range(4) if matrix[r][c] == 0]


def evaluate_board(matrix, empty_cells_count):
    """ 评估函数：分数越高越好，用于搜索的叶子节点 """
    if empty_cells_count == 0:
         
         can_move = False
         for move_key in ALL_MOVES:
             _, changed = simulate_move(matrix, move_key)
             if changed:
                 can_move = True
                 break
         if not can_move:
             return -float('inf') 

    grid_score = 0.0
    smoothness_score = 0.0
    

    for r in range(4):
        for c in range(4):
            value = matrix[r][c]
            if value == 0: continue
            
            
            grid_score += value * WEIGHT_MATRIX[r][c]
            
            
            log_val = math.log2(value)
            
            for dr, dc in [(0, 1), (1, 0)]:
                 nr, nc = r + dr, c + dc
                 if 0 <= nr < 4 and 0 <= nc < 4 and matrix[nr][nc] > 0:
                      log_neighbor = math.log2(matrix[nr][nc])
                      smoothness_score -= abs(log_val - log_neighbor)

    
    empty_score = math.log(empty_cells_count + 1) * EMPTY_CELL_WEIGHT_FACTOR if empty_cells_count > 0 else 0
    
    
    final_score = grid_score + empty_score + smoothness_score * SMOOTHNESS_WEIGHT
    return final_score


def get_search_depth(empty_count):
    """根据空格数动态决定搜索深度，平衡性能与效果"""
    if empty_count >= 7: return 2 
    if empty_count >= 4: return 3 
    
    return 4 


def search(matrix, depth, is_player_turn):
    """ 递归搜索: Expectimax """
    
    
    board_tuple = tuple(map(tuple, matrix)) 
    cache_key = (board_tuple, depth, is_player_turn)
    if cache_key in CACHE:
        return CACHE[cache_key]

    empty_cells = get_empty_cells(matrix)
    
    
    if depth == 0 :
       score = evaluate_board(matrix, len(empty_cells))
       CACHE[cache_key] = score
       return score

    
    if is_player_turn:
        best_score = -float('inf')
        has_valid_move = False
        for move_key in ALL_MOVES:
            new_matrix, changed = simulate_move(matrix, move_key)
            if changed:
                has_valid_move = True
                
                score = search(new_matrix, depth , False) 
                best_score = max(best_score, score)
        
        
        if not has_valid_move:
             best_score = -float('inf') 

        CACHE[cache_key] = best_score
        return best_score

    
    else: 
        num_empty = len(empty_cells)
        if num_empty == 0:
             
             
             score = search(matrix, depth -1 , True) 
             CACHE[cache_key] = score
             return score

        expected_score = 0.0
        
        
        sample_cells = empty_cells 
        
        prob_2 = 0.9 / len(sample_cells)
        prob_4 = 0.1 / len(sample_cells)

        for r, c in sample_cells:
             
             matrix[r][c] = 2
             
             expected_score += prob_2 * search(matrix, depth - 1, True)
             
             
             matrix[r][c] = 4
             expected_score += prob_4 * search(matrix, depth - 1, True)
             
             
             matrix[r][c] = 0 
             
        CACHE[cache_key] = expected_score
        return expected_score


def find_best_move_ai(matrix):
    global CACHE
    
    CACHE.clear() 
    
    best_score = -float('inf')
    best_move = None
    move_scores = {}
    
    empty_count = len(get_empty_cells(matrix))
     
    current_depth = get_search_depth(empty_count)

    if empty_count > 14: 
         return random.choice([Keys.ARROW_DOWN, Keys.ARROW_LEFT])

    for move_key in ALL_MOVES:
        new_matrix, changed = simulate_move(matrix, move_key)
        if changed:
             
            score = search(new_matrix, current_depth, False) 
            move_scores[move_key] = score
            
      
    if not move_scores: return None 

    max_score = max(move_scores.values())
    
    best_moves_candidates = [move for move, score in move_scores.items() if score == max_score and score > -float('inf')]
    if not best_moves_candidates: 
         best_moves_candidates = [move for move, score in move_scores.items() if score == max_score]

    if len(best_moves_candidates) > 1:
        best_moves_candidates.sort(key=lambda k: MOVE_PREFERENCE.get(k, 99))
       
    best_move = best_moves_candidates[0] if best_moves_candidates else None
    
    
    
    return best_move




def setup_driver():
    print("正在设置浏览器驱动...")
    options = webdriver.ChromeOptions()
    
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--log-level=3") 
    options.add_argument("--window-size=700,850")
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        print("浏览器驱动设置完成。")
        return driver
    except Exception as e:
        print(f"\n!!! 驱动设置失败 !!! 错误详情: {e}\n")
        sys.exit(1)


def get_score(driver):
    try:
        score_element = driver.find_element(By.CSS_SELECTOR, ".score-container")
        score_text = score_element.text.split('\n')[0].strip()
        return int(score_text) if score_text.isdigit() else 0
    except (NoSuchElementException, ValueError, StaleElementReferenceException, IndexError): return 0      
def is_game_over(driver):
    try:
       game_over_element = driver.find_element(By.CSS_SELECTOR, ".game-message.game-over")
       return game_over_element.is_displayed() and 'game-over' in game_over_element.get_attribute('class')
    except (NoSuchElementException, StaleElementReferenceException) : return False
def retry_game(driver, wait):
     try:
        print("\n尝试重新开始游戏...")
        retry_button = wait.until( EC.element_to_be_clickable((By.CSS_SELECTOR, "a.retry-button")))
        driver.execute_script("arguments[0].click();", retry_button)
        time.sleep(0.8) 
        print("游戏已重新开始。")
        return True
     except (TimeoutException, StaleElementReferenceException):
        print("未找到或无法点击 'Try Again' 按钮。")
        return False      
def get_board_matrix(driver):
    matrix = [[0] * 4 for _ in range(4)]
    retries = 2
    for attempt in range(retries):
        try:
            tiles = driver.find_elements(By.CSS_SELECTOR, ".tile-container .tile")
            current_max_values = {} 
            for tile in tiles:
                 try: 
                    classes = tile.get_attribute("class").split()
                 except StaleElementReferenceException:
                     continue 
                 value, col, row = 0, -1, -1
                 for cls in classes:
                     if cls.startswith("tile-position-"):
                         parts = cls.split('-')
                         if len(parts) == 4: col, row = int(parts[2]) - 1, int(parts[3]) - 1 
                     elif cls.startswith("tile-") and cls[5:].isdigit(): value = int(cls.split('-')[1])
                 if 0 <= row < 4 and 0 <= col < 4 and value > 0 :
                      pos_key = (row, col)
                      if value > current_max_values.get(pos_key, 0): current_max_values[pos_key] = value
            for (r,c), val in current_max_values.items(): matrix[r][c] = val
            return matrix 
        except StaleElementReferenceException:
              if attempt < retries -1: time.sleep(0.03) 
        except Exception: pass 
    return matrix 
last_printed_matrix_str = ""
def print_matrix(matrix, score, move_str=""):
     global last_printed_matrix_str
     current_matrix_str = str(matrix) + str(score) + move_str
     if current_matrix_str == last_printed_matrix_str: return
     last_printed_matrix_str = current_matrix_str
     print("\033c", end="") 
     header = f" SCORE: {score:<8} | MOVE: {move_str:<6} "
     width = 29
     print("=" * width)
     print(f"|{header:^{width-2}}|")
     print("=" * width)
     max_val = 0
     for row in matrix:
        print("|", end="")
        for cell in row:
            if cell > max_val : max_val = cell
            cell_str = str(cell) if cell > 0 else '.'
            print(f"{cell_str:^6}|", end="") 
        print("\n" + "-" * width)
     print(f"| MAX TILE: {max_val:<15} |")
     print("=" * width)


def play_game(driver, auto_restart):
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    move_count = 0
    last_board_tuple = None
    stuck_count = 0
    try:
        print(f"导航到: {URL}")
        driver.get(URL)
        print("等待游戏加载...")
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "game-container")))
        
        try:
             cookie_btn = driver.find_element(By.CSS_SELECTOR, "button.cookie-banner__ok")
             if cookie_btn.is_displayed(): cookie_btn.click()
             time.sleep(0.5)
        except NoSuchElementException:
             pass
             
        body_element = driver.find_element(By.TAG_NAME, 'body')
        print("游戏加载完毕，AI 开始运行...")
        time.sleep(0.5) 

        while True:
            current_board = get_board_matrix(driver)
            current_board_tuple = tuple(map(tuple, current_board))
            if current_board_tuple == last_board_tuple:
                 stuck_count +=1
            else:
                 stuck_count = 0
                 last_board_tuple = current_board_tuple
            
            if stuck_count > 6: 
                 print("检测到棋盘长时间无变化，强制检查Game Over状态...")
                 time.sleep(0.5) 

            if is_game_over(driver) or stuck_count > 10:
                final_score = get_score(driver)
                board = get_board_matrix(driver)
                print_matrix(board, final_score, "END")
                print(f"\n>>> GAME OVER! <<< | 最终分数: {final_score} | 总步数: {move_count}")
                if auto_restart:
                   if retry_game(driver, wait):
                       move_count, stuck_count, last_board_tuple = 0, 0, None
                       body_element = driver.find_element(By.TAG_NAME, 'body') 
                       continue 
                   else: break 
                else: break 

            current_score = get_score(driver)
            
            move = find_best_move_ai(current_board) 
            move_str = KEY_MAP.get(move, 'WAIT')
            print_matrix(current_board, current_score, move_str)
            print(f"步数: {move_count+1:<4} ")

            if move is None:
                 print("AI 认为无路可走, 等待游戏结束判定...")
                 time.sleep(1) 
                 stuck_count += 2 
                 continue
            try:
               body_element.send_keys(move)
               move_count += 1
            except StaleElementReferenceException:
                body_element = driver.find_element(By.TAG_NAME, 'body')
                body_element.send_keys(move)
                move_count += 1
            time.sleep(ANIMATION_DELAY) 
            
    except TimeoutException: print(f"错误: 等待元素超时 ({WAIT_TIMEOUT}s)。")
    except KeyboardInterrupt: print("\n用户中断脚本。")
    except Exception as e:
         import traceback
         print(f"\n运行中发生未知错误: {e}"); traceback.print_exc()
    finally:
        final_score = get_score(driver)
        print(f"\n脚本结束。当前/最终分数: {final_score}. 将在5秒后关闭浏览器...")
        time.sleep(5); driver.quit(); print("浏览器已关闭。")

if __name__ == "__main__":
    browser_driver = setup_driver()
    if browser_driver: play_game(browser_driver, auto_restart=AUTO_RESTART_GAME) 
    print("--- ALL DONE ---")
