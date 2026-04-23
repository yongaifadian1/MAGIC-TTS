<p align="center">
  <img src="./logo.png" alt="MAGIC-TTS logo" width="220">
</p>

# MAGIC-TTS Demo

中文 | [English](#english)

## 中文

四个场景都使用同一条默认音色，并按照同样的递进方式展开：

1. `v1`：均匀 baseline 时序
2. `v2`：只加入关键停顿
3. `v3`：在关键停顿之上，再拉长局部内容 token

### 场景 1：导航转向

目标：动作词需要被听清，单靠自然合成不够。

文本：`前方路口，左转。`

- `v1` 基线：所有内容字统一设为 `170 ms`  
  [试听](./outputs/controlled_demos/default_voice/01_navigation_turn/01_navigation_turn/v1_baseline_eqdur/gen_target_only.wav)
- `v2` 只加停顿：保持内容字 `170 ms`，并把动作短语前的逗号停顿设为 `260 ms`  
  [试听](./outputs/controlled_demos/default_voice/01_navigation_turn/01_navigation_turn/v2_pause_only_boundary/gen_target_only.wav)
- `v3` 停顿 + 内容拉长：保持逗号停顿 `260 ms`，并把 `左 / 转` 拉长到 `300 ms`  
  [试听](./outputs/controlled_demos/default_voice/01_navigation_turn/01_navigation_turn/v3_pause_plus_char_turn/gen_target_only.wav)
- `spontaneous` 不提供任何 text duration，由 MAGIC-TTS 自行决定时长  
  [试听](./outputs/spontaneous_demos/01_navigation_turn/spontaneous/gen_target_only.wav)

### 场景 2：儿童跟读

目标：教学场景里，目标词的音节时长需要被明确拉出来。

文本：`请跟我读，苹果。`

- `v1` 基线：所有内容字统一设为 `170 ms`  
  [试听](./outputs/controlled_demos/default_voice/02_kids_reading/02_kids_reading/v1_baseline_eqdur/gen_target_only.wav)
- `v2` 只加停顿：保持内容字 `170 ms`，并把目标词前的逗号停顿设为 `260 ms`  
  [试听](./outputs/controlled_demos/default_voice/02_kids_reading/02_kids_reading/v2_pause_only_syllable/gen_target_only.wav)
- `v3` 停顿 + 内容拉长：保持逗号停顿 `260 ms`，并把 `苹 / 果` 拉长到 `300 ms`  
  [试听](./outputs/controlled_demos/default_voice/02_kids_reading/02_kids_reading/v3_pause_plus_char_syllable/gen_target_only.wav)
- `spontaneous` 不提供任何 text duration，由 MAGIC-TTS 自行决定时长  
  [试听](./outputs/spontaneous_demos/02_kids_reading/spontaneous/gen_target_only.wav)

### 场景 3：验证码播报

目标：数字串播报不能只靠自然度，必须控制分组和重点数字时长。

文本：`验证码是三七九，二一八。`

- `v1` 基线：所有内容字统一设为 `170 ms`  
  [试听](./outputs/controlled_demos/default_voice/03_accessibility_code/03_accessibility_code/v1_baseline_eqdur/gen_target_only.wav)
- `v2` 只加停顿：保持内容字 `170 ms`，并把 `3+3` 分组边界处的逗号停顿设为 `260 ms`  
  [试听](./outputs/controlled_demos/default_voice/03_accessibility_code/03_accessibility_code/v2_pause_only_grouped/gen_target_only.wav)
- `v3` 停顿 + 内容拉长：保持逗号停顿 `260 ms`，并把 `三 / 七 / 九 / 二 / 一 / 八` 全部拉长到 `300 ms`  
  [试听](./outputs/controlled_demos/default_voice/03_accessibility_code/03_accessibility_code/v3_pause_plus_char_digits/gen_target_only.wav)
- `spontaneous` 不提供任何 text duration，由 MAGIC-TTS 自行决定时长  
  [试听](./outputs/spontaneous_demos/03_accessibility_code/spontaneous/gen_target_only.wav)

### 场景 4：站点播报

目标：站名前缀和站名本体需要分开，而且站名本体需要被突出。

文本：`前方到站，五山站。`

- `v1` 基线：所有内容字统一设为 `170 ms`  
  [试听](./outputs/controlled_demos/default_voice/04_station_wushanzhan/04_station_wushanzhan/v1_baseline_eqdur/gen_target_only.wav)
- `v2` 只加停顿：保持内容字 `170 ms`，并把站名前的逗号停顿设为 `260 ms`  
  [试听](./outputs/controlled_demos/default_voice/04_station_wushanzhan/04_station_wushanzhan/v2_pause_only_station_boundary/gen_target_only.wav)
- `v3` 停顿 + 内容拉长：保持逗号停顿 `260 ms`，并把 `五 / 山 / 站` 拉长到 `300 ms`  
  [试听](./outputs/controlled_demos/default_voice/04_station_wushanzhan/04_station_wushanzhan/v3_pause_plus_char_station_name/gen_target_only.wav)
- `spontaneous` 不提供任何 text duration，由 MAGIC-TTS 自行决定时长  
  [试听](./outputs/spontaneous_demos/04_station_wushanzhan/spontaneous/gen_target_only.wav)

---

## English

The story is progressive:

1. `v1`: a uniform `170 ms` baseline timing plan  
2. `v2`: keep content at `170 ms` and add a critical `260 ms` pause  
3. `v3`: keep that `260 ms` pause and further lengthen key content tokens to `300 ms`

### 1. Navigation Turn

Text: `前方路口，左转。`

- `v1` baseline: all content tokens set to `170 ms` [listen](./outputs/controlled_demos/default_voice/01_navigation_turn/01_navigation_turn/v1_baseline_eqdur/gen_target_only.wav)
- `v2` pause only: keep content at `170 ms`, set the boundary pause to `260 ms` [listen](./outputs/controlled_demos/default_voice/01_navigation_turn/01_navigation_turn/v2_pause_only_boundary/gen_target_only.wav)
- `v3` pause + content: keep the `260 ms` pause and lengthen `左 / 转` to `300 ms` [listen](./outputs/controlled_demos/default_voice/01_navigation_turn/01_navigation_turn/v3_pause_plus_char_turn/gen_target_only.wav)
- `spontaneous`: do not provide any text duration, letting MAGIC-TTS decide timing on its own [listen](./outputs/spontaneous_demos/01_navigation_turn/spontaneous/gen_target_only.wav)

### 2. Kids Reading

Text: `请跟我读，苹果。`

- `v1` baseline: all content tokens set to `170 ms` [listen](./outputs/controlled_demos/default_voice/02_kids_reading/02_kids_reading/v1_baseline_eqdur/gen_target_only.wav)
- `v2` pause only: keep content at `170 ms`, set the boundary pause to `260 ms` [listen](./outputs/controlled_demos/default_voice/02_kids_reading/02_kids_reading/v2_pause_only_syllable/gen_target_only.wav)
- `v3` pause + content: keep the `260 ms` pause and lengthen `苹 / 果` to `300 ms` [listen](./outputs/controlled_demos/default_voice/02_kids_reading/02_kids_reading/v3_pause_plus_char_syllable/gen_target_only.wav)
- `spontaneous`: do not provide any text duration, letting MAGIC-TTS decide timing on its own [listen](./outputs/spontaneous_demos/02_kids_reading/spontaneous/gen_target_only.wav)

### 3. Accessibility Code

Text: `验证码是三七九，二一八。`

- `v1` baseline: all content tokens set to `170 ms` [listen](./outputs/controlled_demos/default_voice/03_accessibility_code/03_accessibility_code/v1_baseline_eqdur/gen_target_only.wav)
- `v2` pause only: keep content at `170 ms`, set the group pause to `260 ms` [listen](./outputs/controlled_demos/default_voice/03_accessibility_code/03_accessibility_code/v2_pause_only_grouped/gen_target_only.wav)
- `v3` pause + content: keep the `260 ms` pause and lengthen all six digits to `300 ms` [listen](./outputs/controlled_demos/default_voice/03_accessibility_code/03_accessibility_code/v3_pause_plus_char_digits/gen_target_only.wav)
- `spontaneous`: do not provide any text duration, letting MAGIC-TTS decide timing on its own [listen](./outputs/spontaneous_demos/03_accessibility_code/spontaneous/gen_target_only.wav)

### 4. Station Arrival

Text: `前方到站，五山站。`

- `v1` baseline: all content tokens set to `170 ms` [listen](./outputs/controlled_demos/default_voice/04_station_wushanzhan/04_station_wushanzhan/v1_baseline_eqdur/gen_target_only.wav)
- `v2` pause only: keep content at `170 ms`, set the boundary pause to `260 ms` [listen](./outputs/controlled_demos/default_voice/04_station_wushanzhan/04_station_wushanzhan/v2_pause_only_station_boundary/gen_target_only.wav)
- `v3` pause + content: keep the `260 ms` pause and lengthen `五 / 山 / 站` to `300 ms` [listen](./outputs/controlled_demos/default_voice/04_station_wushanzhan/04_station_wushanzhan/v3_pause_plus_char_station_name/gen_target_only.wav)
- `spontaneous`: do not provide any text duration, letting MAGIC-TTS decide timing on its own [listen](./outputs/spontaneous_demos/04_station_wushanzhan/spontaneous/gen_target_only.wav)
