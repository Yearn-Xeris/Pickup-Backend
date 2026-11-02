import pandas as pd
import random
import datetime
from flask import Flask, jsonify
from flask_cors import CORS

# ============================================================
# 第一部分：模拟学生数据及预测逻辑（周末自适应）
# ============================================================
def generate_student_data(num_students=1000):
    students = []
    for i in range(num_students):
        morning = random.choices(["无课", "仅1-2节", "3-4节", "3-5节"], weights=[0.10, 0.20, 0.55, 0.15])[0]
        afternoon = random.choices(["无课", "仅6-7节", "8-9节", "8-10节"], weights=[0.20, 0.25, 0.35, 0.20])[0]
        is_runner = random.random() < 0.10
        is_freshman = random.random() < 0.35
        walking_time = random.randint(5, 15)
        num_packages = random.randint(1, 6)
        package_sizes = random.choices(["小", "中", "大"], weights=[0.6, 0.3, 0.1], k=num_packages)
        has_large_item = "大" in package_sizes
        
        # ⚡ 优化：让包裹在 8:00 到 17:00 之间均匀送达，而不是集中在四个整点
        arr_h = random.randint(8, 16)
        arr_m = random.randint(0, 59)
        arrival_time = f"{arr_h:02d}:{arr_m:02d}"
        
        students.append({
            "student_id": f"S_{i:03d}", "morning": morning, "afternoon": afternoon,
            "is_runner": is_runner, "is_freshman": is_freshman, "walking_time": walking_time,
            "num_packages": num_packages, "package_sizes": package_sizes,
            "has_large_item": has_large_item, "arrival_time": arrival_time
        })
    return pd.DataFrame(students)

def min_to_str(m):
    m = int(m)
    return f"{m//60:02d}:{m%60:02d}"

def predict_checkout_time(row, is_weekend=False):
    m, a, num, large = row['morning'], row['afternoon'], row['num_packages'], row['has_large_item']
    arrive, fresh, runner = row['arrival_time'], row['is_freshman'], row['is_runner']
    
    # ⚡ 修复：自动把 "14:35" 解析成分钟，不再使用写死的字典
    arr_h, arr_m = map(int, arrive.split(':'))
    arrival_minute = arr_h * 60 + arr_m
    
    free_windows = []
    
    if is_weekend:
        # 周末模式：时间分散
        free_windows.append((10*60, 12*60, "周末起床后", 0.8))
        free_windows.append((12*60, 14*60, "周末出校吃午饭", 1.5))
        free_windows.append((14*60, 17*60, "周末下午闲逛", 1.2))
        free_windows.append((17*60, 19*60, "周末出校吃晚饭", 1.5))
        free_windows.append((19*60, 22*60, "周末晚上散步", 1.0))
    else:
        # 工作日模式
        if m == "无课": free_windows.append((9*60, 11*60, "上午专门去取", 1.2))
        if m == "仅1-2节": free_windows.append((9*60+35, 11*60+30, "1-2节后专门去取", 1.3))
        if m == "3-4节": free_windows.append((11*60+25, 13*60, "出校吃午饭顺路取", 1.5))
        if m == "3-5节": free_windows.append((12*60+30, 13*60+30, "出校吃午饭顺路取", 1.3))
        if m in ["无课", "仅1-2节"]: free_windows.append((11*60+30, 13*60, "中午出校吃饭顺路取", 1.2))
        if a == "无课":
            free_windows.append((13*60+30, 15*60, "下午专门去取", 1.0))
            free_windows.append((15*60, 17*60, "下午空闲去取", 0.9))
        if a == "仅6-7节": free_windows.append((15*60+5, 17*60, "15:05下课后专门去取", 1.0))
        if a in ["8-9节", "8-10节"]: free_windows.append((13*60+30, 15*60+5, "6-7节没课专门去取", 1.0))
        if a == "8-9节": free_windows.append((16*60+55, 18*60, "16:55下课后出校顺路", 1.5))
        if a == "8-10节": free_windows.append((17*60+45, 19*60, "17:45下课后出校顺路", 1.4))
        free_windows.append((17*60+30, 19*60, "出校吃晚饭顺路取", 1.2))
        if not fresh:
            free_windows.append((19*60, 20*60+30, "晚间专门去取", 0.7))
            free_windows.append((20*60+30, 22*60, "晚间再取", 0.5))
        else:
            free_windows.append((20*60+30, 22*60, "晚自习后出校取", 0.9))
        if runner and m in ["无课", "仅1-2节"]:
            free_windows.append((9*60+30, 10*60+30, "晨跑后专门取件", 1.0))
    
    # 通用裁剪逻辑
    valid_windows = []
    for w in free_windows:
        start, end, label, weight = w
        if end <= arrival_minute: continue
        valid_windows.append((max(start, arrival_minute), end, label, weight))
    
    if not valid_windows: 
        return "19:00-20:00" if is_weekend else "17:45-18:30"
    
    weights = [w[3] for w in valid_windows]
    chosen = random.choices(valid_windows, weights=weights, k=1)[0]
    start, end, label = chosen[0], chosen[1], chosen[2]
    
    if large or num >= 4:
        start = min(start + 20, end - 15)
        end = min(start + 30, end)
        
    if not is_weekend and m == "3-5节" and start <= 12*60+30:
        start, end = 12*60 + 45, 13*60 + 15
        
    if end <= start: end = start + 20
    return f"{min_to_str(start)}-{min_to_str(end)}"

def generate_pickup_moments(df, is_weekend=False):
    results = []
    for _, row in df.iterrows():
        time_range = predict_checkout_time(row, is_weekend=is_weekend)
        start_str, end_str = time_range.split('-')
        sh, sm = map(int, start_str.split(':'))
        eh, em = map(int, end_str.split(':'))
        start_min, end_min = sh*60+sm, eh*60+em
        checkout_minute = random.randint(start_min, end_min)
        
        if row['has_large_item'] or row['num_packages'] >= 4: search_time = random.randint(10, 15)
        elif row['num_packages'] >= 3: search_time = random.randint(7, 10)
        else: search_time = random.randint(5, 8)
        
        results.append({
            'student_id': row['student_id'],
            'enter_minute': checkout_minute - search_time,
            'checkout_minute': checkout_minute
        })
    return pd.DataFrame(results)

def calculate_people_at(pickup_df, minute):
    mask = (pickup_df['enter_minute'] <= minute) & (pickup_df['checkout_minute'] >= minute)
    return int(mask.sum())

def get_smart_advice(current_people, future_slots):
    next_20_min_max = max([slot['count'] for slot in future_slots[:2]]) if len(future_slots) >= 2 else current_people
    
    if current_people < 20:
        status = "🟢 正常"
    elif current_people < 50:
        status = "🟡 轻微拥挤"
    else:
        status = "🔴 爆满"

    if current_people < 20 and next_20_min_max > 50:
        advice = "⚠️ 当前人少，但马上要迎来下课/饭点大爆发！建议立刻去取，晚了就要排长队了！"
    elif current_people < 20 and next_20_min_max < 20:
        advice = "当前人很少，可以随时去取。"
    elif current_people >= 50:
        if next_20_min_max < current_people - 10:
            advice = "现在虽然人多，但马上就会散去，建议在附近等10分钟再进。"
        else:
            advice = "驿站爆满！建议避开这个时段，30分钟后再来。"
    elif 20 <= current_people < 50:
        if next_20_min_max > 80:
            advice = "⚠️ 抓紧时间！现在人还不算最多，10分钟后将彻底爆满！"
        else:
            advice = "当前有少量排队，如果赶时间建议直接去，不着急可以等会儿。"
            
    return status, advice

# ============================================================
# 初始化数据：生成两套数据备用
# ============================================================
print("正在生成模拟数据，请稍候...")
# 建议人数设置在 1000 左右，图表会比较好看
df_students = generate_student_data(1000) 

print("-> 正在生成【工作日】数据...")
pickup_df_workday = generate_pickup_moments(df_students, is_weekend=False)

print("-> 正在生成【周末】数据...")
pickup_df_weekend = generate_pickup_moments(df_students, is_weekend=True)

print("✅ 数据生成完毕！正在启动 Flask 服务器...")

# ============================================================
# 第二部分：启动 Flask API
# ============================================================
app = Flask(__name__)
CORS(app)
app.json.ensure_ascii = False

@app.route('/api/traffic')
def get_traffic():
    now = datetime.datetime.now()
    current_minute = now.hour * 60 + now.minute
    is_weekend = now.weekday() >= 5
    
    # 动态切换数据集
    current_pickup_df = pickup_df_weekend if is_weekend else pickup_df_workday
    
    current_people = calculate_people_at(current_pickup_df, current_minute)
    
    # 计算未来6个时段
    future_slots = []
    for i in range(0, 60, 10):
        slot_start = current_minute + i
        slot_end = slot_start + 10
        people_series = [calculate_people_at(current_pickup_df, m) for m in range(slot_start, slot_end)]
        avg_people = round(sum(people_series) / len(people_series), 1) if people_series else 0
        future_slots.append({
            'time': min_to_str(slot_start),
            'count': avg_people
        })
        
    # 调用智能建议函数
    status_text, advice = get_smart_advice(current_people, future_slots)
        
    return jsonify({
        "current_people": current_people,
        "status_text": status_text,
        "advice": advice,
        "future_slots": future_slots,
        "is_weekend": is_weekend 
    })

if __name__ == '__main__':
    app.run(debug=False)