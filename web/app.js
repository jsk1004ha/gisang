const API_BASE = (typeof window !== "undefined" && window.API_BASE) || "";

const HORIZONS = [
  { key: "d1", day: 1, label: "내일 D+1", shortLabel: "D+1" },
  { key: "d2", day: 2, label: "D+2", shortLabel: "D+2" },
  { key: "d3", day: 3, label: "D+3", shortLabel: "D+3" },
  { key: "d5", day: 5, label: "D+5", shortLabel: "D+5" },
];

const VARIABLE_DEFS = {
  temperature: {
    key: "temperature",
    label: "기온",
    legend: "기온 (°C)",
    unit: "°C",
    prefix: "temperature",
    statusPrefix: "temperature",
    className: "metric-temperature",
    ticks: ["30", "25", "20", "15", "10", "5", "0", "-5"],
  },
  humidity: {
    key: "humidity",
    label: "습도 beta",
    legend: "습도 (%)",
    unit: "%",
    prefix: "humidity",
    statusPrefix: "humidity",
    className: "metric-humidity",
    ticks: ["90", "80", "70", "60", "50", "40", "30", "20"],
  },
  precip_probability: {
    key: "precip_probability",
    label: "강수확률",
    legend: "강수확률 (%)",
    unit: "%",
    prefix: "precip_probability",
    statusPrefix: "precip_probability",
    className: "metric-precip",
    ticks: ["100", "80", "60", "40", "20", "10", "5", "0"],
  },
  wind: {
    key: "wind",
    label: "바람",
    legend: "바람 (m/s)",
    unit: "m/s",
    prefix: "wind",
    statusPrefix: "wind",
    className: "metric-wind",
    ticks: ["12", "10", "8", "6", "4", "2", "1", "0"],
  },
  cloud: {
    key: "cloud",
    label: "구름",
    legend: "구름량 (%)",
    unit: "%",
    prefix: "cloud",
    statusPrefix: "cloud",
    className: "metric-cloud",
    ticks: ["100", "80", "60", "40", "20", "10", "5", "0"],
  },
  weather_code: {
    key: "weather_code",
    label: "날씨상태",
    legend: "날씨 상태",
    unit: "",
    prefix: "weather_code",
    statusPrefix: "weather_code",
    className: "metric-weather",
    ticks: ["비", "흐림", "구름", "맑음"],
  },
};

const SOURCE_LABELS = {
  ai_mos_model: "AI 예측",
  ai_mos: "AI 예측",
  temperature_ai_mos: "AI 예측",
  ai_mos_model_beta: "AI beta",
  ai_mos_beta: "AI beta",
  humidity_ai_mos_beta: "AI beta",
  ai_beta: "AI beta",
  nwp_direct: "NWP direct",
  gfs_direct: "NWP direct",
  kma_direct: "KMA direct",
  kma_forecast_direct: "KMA direct",
  rule_based_beta: "규칙 기반 beta",
  unavailable: "준비 중",
  unknown: "준비 중",
};


const SOURCE_FIELD_NAMES = [
  "temperature_source",
  "temperature_status",
  "temperature_confidence",
  "humidity_source",
  "humidity_status",
  "humidity_confidence",
  "precip_probability_source",
  "precip_probability_status",
  "precip_probability_confidence",
  "wind_source",
  "wind_status",
  "wind_confidence",
  "cloud_source",
  "cloud_status",
  "cloud_confidence",
  "weather_code_source",
  "weather_code_status",
  "weather_code_confidence",
];

const SOURCE_TITLES = {
  rule_based_beta: "rule-based beta",
  unavailable: "unavailable",
};

const FALLBACK_COORDINATES = {
  "90": { x: 58.1, y: 9.5, name: "속초" },
  SEOUL: { x: 35.2, y: 21.2, name: "서울" },
  "93": { x: 46.5, y: 14.7, name: "북춘천" },
  "95": { x: 40.1, y: 11.2, name: "철원" },
  "98": { x: 36.6, y: 15.5, name: "동두천" },
  "99": { x: 32.4, y: 15.8, name: "파주" },
  "100": { x: 60.3, y: 19.4, name: "대관령" },
  "101": { x: 46.2, y: 15.5, name: "춘천" },
  "102": { x: 14.0, y: 14.2, name: "백령도" },
  "104": { x: 62.2, y: 17.2, name: "북강릉" },
  "105": { x: 62.7, y: 18.1, name: "강릉" },
  "106": { x: 66.1, y: 22.3, name: "동해" },
  "108": { x: 35.2, y: 21.2, name: "서울" },
  "112": { x: 30.4, y: 22.8, name: "인천" },
  "114": { x: 49.2, y: 25.2, name: "원주" },
  "115": { x: 86.0, y: 22.7, name: "울릉도" },
  "119": { x: 35.5, y: 26.6, name: "수원" },
  "121": { x: 56.5, y: 27.9, name: "영월" },
  "127": { x: 49.3, y: 31.5, name: "충주" },
  "129": { x: 28.5, y: 34.9, name: "서산" },
  "130": { x: 70.2, y: 31.2, name: "울진" },
  "131": { x: 42.0, y: 37.3, name: "청주" },
  "133": { x: 41.0, y: 41.9, name: "대전" },
  "135": { x: 49.9, y: 44.5, name: "추풍령" },
  "136": { x: 60.1, y: 38.4, name: "안동" },
  "137": { x: 52.2, y: 41.2, name: "상주" },
  "138": { x: 69.7, y: 47.7, name: "포항" },
  "140": { x: 32.3, y: 48.2, name: "군산" },
  "143": { x: 59.3, y: 50.4, name: "대구" },
  "146": { x: 37.4, y: 51.0, name: "전주" },
  "152": { x: 69.1, y: 55.5, name: "울산" },
  "156": { x: 34.2, y: 62.5, name: "광주" },
  "159": { x: 64.7, y: 63.7, name: "부산" },
  "162": { x: 56.2, y: 68.2, name: "통영" },
  "165": { x: 26.9, y: 68.7, name: "목포" },
  "168": { x: 46.3, y: 70.0, name: "여수" },
  "170": { x: 31.5, y: 75.9, name: "완도" },
  "184": { x: 29.0, y: 91.1, name: "제주" },
  "185": { x: 23.8, y: 92.0, name: "고산" },
  "189": { x: 29.5, y: 92.0, name: "서귀포" },
  "192": { x: 50.6, y: 62.7, name: "진주" },
  "201": { x: 27.8, y: 18.8, name: "강화" },
  "202": { x: 42.8, y: 22.6, name: "양평" },
  "203": { x: 42.6, y: 26.5, name: "이천" },
  "221": { x: 52.8, y: 28.3, name: "제천" },
  "232": { x: 39.9, y: 35.1, name: "천안" },
  "235": { x: 29.4, y: 42.6, name: "보령" },
  "238": { x: 42.6, y: 46.5, name: "금산" },
  "243": { x: 31.7, y: 52.9, name: "부안" },
  "244": { x: 39.8, y: 55.0, name: "임실" },
  "245": { x: 33.4, y: 55.8, name: "정읍" },
  "247": { x: 41.4, y: 58.3, name: "남원" },
  "248": { x: 43.1, y: 54.2, name: "장수" },
  "251": { x: 30.0, y: 59.5, name: "고창" },
  "253": { x: 41.0, y: 65.2, name: "순천" },
  "255": { x: 59.6, y: 61.6, name: "북창원" },
  "257": { x: 64.6, y: 60.2, name: "양산" },
  "258": { x: 38.7, y: 69.6, name: "보성" },
  "259": { x: 32.6, y: 71.6, name: "강진" },
  "260": { x: 34.6, y: 70.9, name: "장흥" },
  "261": { x: 29.6, y: 73.2, name: "해남" },
  "262": { x: 39.7, y: 72.1, name: "고흥" },
  "263": { x: 54.1, y: 60.0, name: "의령" },
  "264": { x: 46.4, y: 56.7, name: "함양" },
  "266": { x: 45.6, y: 66.5, name: "광양" },
  "271": { x: 63.1, y: 32.0, name: "봉화" },
  "272": { x: 57.4, y: 33.2, name: "영주" },
  "273": { x: 52.1, y: 37.5, name: "문경" },
  "277": { x: 70.1, y: 39.1, name: "영덕" },
  "278": { x: 59.8, y: 42.1, name: "의성" },
  "279": { x: 54.6, y: 46.0, name: "구미" },
  "281": { x: 63.6, y: 48.7, name: "영천" },
  "284": { x: 48.7, y: 54.0, name: "거창" },
  "285": { x: 52.4, y: 55.8, name: "합천" },
  "288": { x: 60.6, y: 57.0, name: "밀양" },
  "289": { x: 48.3, y: 58.4, name: "산청" },
  "294": { x: 58.6, y: 67.4, name: "거제" },
  "295": { x: 48.9, y: 68.7, name: "남해" },
};

const REGION_COORDINATES = {
  수도권: { x: 47, y: 24 },
  강원권: { x: 65, y: 24 },
  충청권: { x: 50, y: 42 },
  전라권: { x: 42, y: 65 },
  경상권: { x: 65, y: 65 },
  제주권: { x: 38, y: 91 },
};

const FALLBACK_LATLON = {
  "90": { lat: 38.25085, lon: 128.56473 },
  SEOUL: { lat: 37.57142, lon: 126.96580 },
  "93": { lat: 37.94738, lon: 127.75443 },
  "95": { lat: 38.14787, lon: 127.30420 },
  "98": { lat: 37.90188, lon: 127.06070 },
  "99": { lat: 37.88589, lon: 126.76648 },
  "100": { lat: 37.67713, lon: 128.71834 },
  "101": { lat: 37.90262, lon: 127.73570 },
  "102": { lat: 37.97396, lon: 124.71237 },
  "104": { lat: 37.80456, lon: 128.85535 },
  "105": { lat: 37.75147, lon: 128.89099 },
  "106": { lat: 37.50709, lon: 129.12433 },
  "108": { lat: 37.57142, lon: 126.96580 },
  "112": { lat: 37.47772, lon: 126.62490 },
  "114": { lat: 37.33749, lon: 127.94659 },
  "115": { lat: 37.48129, lon: 130.89864 },
  "119": { lat: 37.25746, lon: 126.98300 },
  "121": { lat: 37.18126, lon: 128.45743 },
  "127": { lat: 36.97045, lon: 127.95250 },
  "129": { lat: 36.77658, lon: 126.49390 },
  "130": { lat: 36.99176, lon: 129.41278 },
  "131": { lat: 36.63924, lon: 127.44066 },
  "133": { lat: 36.37199, lon: 127.37210 },
  "135": { lat: 36.22025, lon: 127.99458 },
  "136": { lat: 36.57293, lon: 128.70734 },
  "137": { lat: 36.40837, lon: 128.15741 },
  "138": { lat: 36.03201, lon: 129.38002 },
  "140": { lat: 36.00530, lon: 126.76135 },
  "143": { lat: 35.87797, lon: 128.65296 },
  "146": { lat: 35.84092, lon: 127.11718 },
  "152": { lat: 35.58237, lon: 129.33469 },
  "156": { lat: 35.17294, lon: 126.89156 },
  "159": { lat: 35.10468, lon: 129.03203 },
  "162": { lat: 34.84541, lon: 128.43561 },
  "165": { lat: 34.81732, lon: 126.38151 },
  "168": { lat: 34.73929, lon: 127.74063 },
  "170": { lat: 34.39590, lon: 126.70182 },
  "184": { lat: 33.51411, lon: 126.52969 },
  "185": { lat: 33.29382, lon: 126.16283 },
  "189": { lat: 33.24616, lon: 126.56530 },
  "192": { lat: 35.16378, lon: 128.04004 },
  "201": { lat: 37.70739, lon: 126.44634 },
  "202": { lat: 37.48863, lon: 127.49446 },
  "203": { lat: 37.26399, lon: 127.48421 },
  "221": { lat: 37.15928, lon: 128.19434 },
  "232": { lat: 36.76217, lon: 127.29282 },
  "235": { lat: 36.32724, lon: 126.55744 },
  "238": { lat: 36.10563, lon: 127.48175 },
  "243": { lat: 35.72961, lon: 126.71657 },
  "244": { lat: 35.61203, lon: 127.28556 },
  "245": { lat: 35.56337, lon: 126.83904 },
  "247": { lat: 35.42130, lon: 127.39652 },
  "248": { lat: 35.65696, lon: 127.52031 },
  "251": { lat: 35.34824, lon: 126.59900 },
  "253": { lat: 35.02040, lon: 127.36940 },
  "255": { lat: 35.22655, lon: 128.67260 },
  "257": { lat: 35.30737, lon: 129.02010 },
  "258": { lat: 34.76335, lon: 127.21226 },
  "259": { lat: 34.64457, lon: 126.78408 },
  "260": { lat: 34.68886, lon: 126.91951 },
  "261": { lat: 34.55375, lon: 126.56907 },
  "262": { lat: 34.61826, lon: 127.27572 },
  "263": { lat: 35.32258, lon: 128.28812 },
  "264": { lat: 35.51138, lon: 127.74538 },
  "266": { lat: 34.94340, lon: 127.69140 },
  "271": { lat: 36.94361, lon: 128.91449 },
  "272": { lat: 36.87183, lon: 128.51687 },
  "273": { lat: 36.62727, lon: 128.14879 },
  "277": { lat: 36.53337, lon: 129.40926 },
  "278": { lat: 36.35610, lon: 128.68864 },
  "279": { lat: 36.13055, lon: 128.32056 },
  "281": { lat: 35.97742, lon: 128.95140 },
  "284": { lat: 35.66739, lon: 127.90990 },
  "285": { lat: 35.56505, lon: 128.16994 },
  "288": { lat: 35.49147, lon: 128.74413 },
  "289": { lat: 35.41300, lon: 127.87910 },
  "294": { lat: 34.88818, lon: 128.60459 },
  "295": { lat: 34.81662, lon: 127.92641 },
};

const REGION_LATLON = {
  수도권: { lat: 37.45, lon: 127.02 },
  강원권: { lat: 37.55, lon: 128.2 },
  충청권: { lat: 36.5, lon: 127.4 },
  전라권: { lat: 35.4, lon: 126.9 },
  경상권: { lat: 35.8, lon: 128.7 },
  제주권: { lat: 33.5, lon: 126.53 },
};

const KOREA_MAP_BOUNDS = [
  [33.0, 124.5],
  [38.9, 131.2],
];

const PRIMARY_STATION_IDS = new Set([
  "108", // 서울
  "112", // 인천
  "133", // 대전
  "131", // 청주
  "143", // 대구
  "159", // 부산
  "152", // 울산
  "156", // 광주
  "146", // 전주
  "184", // 제주
  "105", // 강릉
  "165", // 목포
  "168", // 여수
  "115", // 울릉도
]);

const PRIMARY_STATION_NAMES = new Set([
  "서울",
  "인천",
  "대전",
  "청주",
  "대구",
  "부산",
  "울산",
  "광주",
  "전주",
  "제주",
  "강릉",
  "목포",
  "여수",
  "울릉도",
]);

const SECONDARY_STATION_IDS = new Set([
  ...PRIMARY_STATION_IDS,
  "90", // 속초
  "93", // 북춘천
  "95", // 철원
  "98", // 동두천
  "99", // 파주
  "101", // 춘천
  "102", // 백령도
  "106", // 동해
  "119", // 수원
  "127", // 충주
  "129", // 서산
  "130", // 울진
  "136", // 안동
  "138", // 포항
  "140", // 군산
  "162", // 통영
  "170", // 완도
  "189", // 서귀포
  "192", // 진주
]);

const EMPTY_STATION_ANCHORS = [
  { station_id: "90", station_name: "속초", region_class: "강원권", lat: 38.25085, lon: 128.56473 },
  { station_id: "93", station_name: "북춘천", region_class: "강원권", lat: 37.94738, lon: 127.75443 },
  { station_id: "95", station_name: "철원", region_class: "강원권", lat: 38.14787, lon: 127.30420 },
  { station_id: "98", station_name: "동두천", region_class: "수도권", lat: 37.90188, lon: 127.06070 },
  { station_id: "99", station_name: "파주", region_class: "수도권", lat: 37.88589, lon: 126.76648 },
  { station_id: "100", station_name: "대관령", region_class: "강원권", lat: 37.67713, lon: 128.71834 },
  { station_id: "101", station_name: "춘천", region_class: "강원권", lat: 37.90262, lon: 127.73570 },
  { station_id: "102", station_name: "백령도", region_class: "수도권", lat: 37.97396, lon: 124.71237 },
  { station_id: "104", station_name: "북강릉", region_class: "강원권", lat: 37.80456, lon: 128.85535 },
  { station_id: "105", station_name: "강릉", region_class: "강원권", lat: 37.75147, lon: 128.89099 },
  { station_id: "106", station_name: "동해", region_class: "강원권", lat: 37.50709, lon: 129.12433 },
  { station_id: "108", station_name: "서울", region_class: "수도권", lat: 37.57142, lon: 126.96580 },
  { station_id: "112", station_name: "인천", region_class: "수도권", lat: 37.47772, lon: 126.62490 },
  { station_id: "114", station_name: "원주", region_class: "강원권", lat: 37.33749, lon: 127.94659 },
  { station_id: "115", station_name: "울릉도", region_class: "경상권", lat: 37.48129, lon: 130.89864 },
  { station_id: "119", station_name: "수원", region_class: "수도권", lat: 37.25746, lon: 126.98300 },
  { station_id: "121", station_name: "영월", region_class: "강원권", lat: 37.18126, lon: 128.45743 },
  { station_id: "127", station_name: "충주", region_class: "충청권", lat: 36.97045, lon: 127.95250 },
  { station_id: "129", station_name: "서산", region_class: "충청권", lat: 36.77658, lon: 126.49390 },
  { station_id: "130", station_name: "울진", region_class: "경상권", lat: 36.99176, lon: 129.41278 },
  { station_id: "131", station_name: "청주", region_class: "충청권", lat: 36.63924, lon: 127.44066 },
  { station_id: "133", station_name: "대전", region_class: "충청권", lat: 36.37199, lon: 127.37210 },
  { station_id: "135", station_name: "추풍령", region_class: "충청권", lat: 36.22025, lon: 127.99458 },
  { station_id: "136", station_name: "안동", region_class: "경상권", lat: 36.57293, lon: 128.70734 },
  { station_id: "137", station_name: "상주", region_class: "경상권", lat: 36.40837, lon: 128.15741 },
  { station_id: "138", station_name: "포항", region_class: "경상권", lat: 36.03201, lon: 129.38002 },
  { station_id: "140", station_name: "군산", region_class: "전라권", lat: 36.00530, lon: 126.76135 },
  { station_id: "143", station_name: "대구", region_class: "경상권", lat: 35.87797, lon: 128.65296 },
  { station_id: "146", station_name: "전주", region_class: "전라권", lat: 35.84092, lon: 127.11718 },
  { station_id: "152", station_name: "울산", region_class: "경상권", lat: 35.58237, lon: 129.33469 },
  { station_id: "156", station_name: "광주", region_class: "전라권", lat: 35.17294, lon: 126.89156 },
  { station_id: "159", station_name: "부산", region_class: "경상권", lat: 35.10468, lon: 129.03203 },
  { station_id: "162", station_name: "통영", region_class: "경상권", lat: 34.84541, lon: 128.43561 },
  { station_id: "165", station_name: "목포", region_class: "전라권", lat: 34.81732, lon: 126.38151 },
  { station_id: "168", station_name: "여수", region_class: "전라권", lat: 34.73929, lon: 127.74063 },
  { station_id: "170", station_name: "완도", region_class: "전라권", lat: 34.39590, lon: 126.70182 },
  { station_id: "184", station_name: "제주", region_class: "제주권", lat: 33.51411, lon: 126.52969 },
  { station_id: "185", station_name: "고산", region_class: "제주권", lat: 33.29382, lon: 126.16283 },
  { station_id: "189", station_name: "서귀포", region_class: "제주권", lat: 33.24616, lon: 126.56530 },
  { station_id: "192", station_name: "진주", region_class: "경상권", lat: 35.16378, lon: 128.04004 },
  { station_id: "201", station_name: "강화", region_class: "수도권", lat: 37.70739, lon: 126.44634 },
  { station_id: "202", station_name: "양평", region_class: "수도권", lat: 37.48863, lon: 127.49446 },
  { station_id: "203", station_name: "이천", region_class: "수도권", lat: 37.26399, lon: 127.48421 },
  { station_id: "221", station_name: "제천", region_class: "충청권", lat: 37.15928, lon: 128.19434 },
  { station_id: "232", station_name: "천안", region_class: "충청권", lat: 36.76217, lon: 127.29282 },
  { station_id: "235", station_name: "보령", region_class: "충청권", lat: 36.32724, lon: 126.55744 },
  { station_id: "238", station_name: "금산", region_class: "충청권", lat: 36.10563, lon: 127.48175 },
  { station_id: "243", station_name: "부안", region_class: "전라권", lat: 35.72961, lon: 126.71657 },
  { station_id: "244", station_name: "임실", region_class: "전라권", lat: 35.61203, lon: 127.28556 },
  { station_id: "245", station_name: "정읍", region_class: "전라권", lat: 35.56337, lon: 126.83904 },
  { station_id: "247", station_name: "남원", region_class: "전라권", lat: 35.42130, lon: 127.39652 },
  { station_id: "248", station_name: "장수", region_class: "전라권", lat: 35.65696, lon: 127.52031 },
  { station_id: "251", station_name: "고창", region_class: "전라권", lat: 35.34824, lon: 126.59900 },
  { station_id: "253", station_name: "순천", region_class: "전라권", lat: 35.02040, lon: 127.36940 },
  { station_id: "255", station_name: "북창원", region_class: "경상권", lat: 35.22655, lon: 128.67260 },
  { station_id: "257", station_name: "양산", region_class: "경상권", lat: 35.30737, lon: 129.02010 },
  { station_id: "258", station_name: "보성", region_class: "전라권", lat: 34.76335, lon: 127.21226 },
  { station_id: "259", station_name: "강진", region_class: "전라권", lat: 34.64457, lon: 126.78408 },
  { station_id: "260", station_name: "장흥", region_class: "전라권", lat: 34.68886, lon: 126.91951 },
  { station_id: "261", station_name: "해남", region_class: "전라권", lat: 34.55375, lon: 126.56907 },
  { station_id: "262", station_name: "고흥", region_class: "전라권", lat: 34.61826, lon: 127.27572 },
  { station_id: "263", station_name: "의령", region_class: "경상권", lat: 35.32258, lon: 128.28812 },
  { station_id: "264", station_name: "함양", region_class: "경상권", lat: 35.51138, lon: 127.74538 },
  { station_id: "266", station_name: "광양", region_class: "전라권", lat: 34.94340, lon: 127.69140 },
  { station_id: "271", station_name: "봉화", region_class: "경상권", lat: 36.94361, lon: 128.91449 },
  { station_id: "272", station_name: "영주", region_class: "경상권", lat: 36.87183, lon: 128.51687 },
  { station_id: "273", station_name: "문경", region_class: "경상권", lat: 36.62727, lon: 128.14879 },
  { station_id: "277", station_name: "영덕", region_class: "경상권", lat: 36.53337, lon: 129.40926 },
  { station_id: "278", station_name: "의성", region_class: "경상권", lat: 36.35610, lon: 128.68864 },
  { station_id: "279", station_name: "구미", region_class: "경상권", lat: 36.13055, lon: 128.32056 },
  { station_id: "281", station_name: "영천", region_class: "경상권", lat: 35.97742, lon: 128.95140 },
  { station_id: "284", station_name: "거창", region_class: "경상권", lat: 35.66739, lon: 127.90990 },
  { station_id: "285", station_name: "합천", region_class: "경상권", lat: 35.56505, lon: 128.16994 },
  { station_id: "288", station_name: "밀양", region_class: "경상권", lat: 35.49147, lon: 128.74413 },
  { station_id: "289", station_name: "산청", region_class: "경상권", lat: 35.41300, lon: 127.87910 },
  { station_id: "294", station_name: "거제", region_class: "경상권", lat: 34.88818, lon: 128.60459 },
  { station_id: "295", station_name: "남해", region_class: "경상권", lat: 34.81662, lon: 127.92641 },
];

const state = {
  latest: {},
  modelStatus: {},
  allPoints: [],
  selectedStationId: "",
  selectedHorizon: 1,
  selectedVariable: "temperature",
  stationCache: new Map(),
  selectedStationPoints: [],
  selectedDaily: [],
  stationDataWarning: "",
};

const leafletState = {
  map: null,
  markers: [],
  initialFitDone: false,
  updatingBounds: false,
};

async function fetchJson(path) {
  const response = await fetch(API_BASE + path);
  if (!response.ok) {
    throw new Error(path);
  }
  return response.json();
}

async function fetchFirstJson(paths = []) {
  let lastError = null;
  for (const path of paths) {
    try {
      return await fetchJson(path);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("no fetch path");
}

function embeddedForecastData() {
  if (typeof window !== "undefined" && window.GISANG_FORECAST_DATA) {
    return window.GISANG_FORECAST_DATA;
  }
  return null;
}

function embeddedModelStatusData() {
  if (typeof window !== "undefined" && window.GISANG_MODEL_STATUS_DATA) {
    return window.GISANG_MODEL_STATUS_DATA;
  }
  return null;
}

async function fetchLatestForecast() {
  try {
    return await fetchJson("/api/forecast/latest");
  } catch (apiError) {
    try {
      const run = await fetchFirstJson([
        "data/forecasts/latest/forecast_run.json",
        "/data/forecasts/latest/forecast_run.json",
        "../data/forecasts/latest/forecast_run.json",
      ]);
      const points = await fetchFirstJson([
        "data/forecasts/latest/forecast_points.json",
        "/data/forecasts/latest/forecast_points.json",
        "../data/forecasts/latest/forecast_points.json",
      ]);
      return {
        run,
        warnings: run.warnings || [],
        points: Array.isArray(points) ? points : points.points || points.forecast_points || [],
        beta_targets: run.beta_targets || {},
        api_error: apiError.message || "api unavailable",
      };
    } catch (staticError) {
      const embedded = embeddedForecastData();
      if (embedded) {
        return {
          ...embedded,
          api_error: apiError.message || "api unavailable",
          static_error: staticError.message || "static artifact unavailable",
        };
      }
      throw staticError;
    }
  }
}

async function fetchModelStatus() {
  try {
    return await fetchJson("/api/model/status");
  } catch (apiError) {
    try {
      const embedded = embeddedModelStatusData();
      if (embedded) {
        return embedded;
      }
      const manifest = await fetchFirstJson([
        "data/artifacts/g030_production_freeze_final/production_model_manifest.json",
        "/data/artifacts/g030_production_freeze_final/production_model_manifest.json",
        "../data/artifacts/g030_production_freeze_final/production_model_manifest.json",
      ]);
      return {
        ...manifest,
        temp_rmse: manifest.temp_rmse || manifest.temperature_rmse,
        temperature_rmse: manifest.temperature_rmse || manifest.temp_rmse,
        v4_c_gate_status: manifest.v4_c_gate_status || manifest.v4c_gate_status,
        warnings: manifest.site_caveats || [],
      };
    } catch (manifestError) {
      return {
        warnings: [
          `모델 성능 요약 파일이 없습니다: ${manifestError.message || apiError.message || "unknown"}`,
        ],
      };
    }
  }
}

function createElement(tagName, className = "", text = "") {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== "") {
    element.textContent = text;
  }
  return element;
}

function appendTextElement(parent, tagName, text, className = "") {
  const element = createElement(tagName, className, String(text));
  parent.appendChild(element);
  return element;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeLatestPayload(payload = {}) {
  const run = payload.run || payload.forecast_run || {};
  const points = Array.isArray(payload.points)
    ? payload.points
    : Array.isArray(payload.forecast_points)
      ? payload.forecast_points
      : [];
  return { ...payload, run, points };
}

function normalizeSource(source) {
  const normalized = String(source || "unavailable").trim().toLowerCase();
  return normalized || "unavailable";
}

function sourceLabel(source) {
  const normalized = normalizeSource(source);
  return SOURCE_LABELS[normalized] || normalized.replaceAll("_", " ");
}

function sourceTitle(source) {
  const normalized = normalizeSource(source);
  return SOURCE_TITLES[normalized] || sourceLabel(normalized);
}

function sourceMetadata(point = {}, prefix) {
  const sourceKey = `${prefix}_source`;
  const statusKey = `${prefix}_status`;
  const confidenceKey = `${prefix}_confidence`;
  const hasExplicitSource = Object.prototype.hasOwnProperty.call(point, sourceKey);
  const hasExplicitStatus = Object.prototype.hasOwnProperty.call(point, statusKey);
  const source = hasExplicitSource ? normalizeSource(point[sourceKey]) : "unavailable";
  const status = hasExplicitStatus ? String(point[statusKey] || "unknown") : "unknown";
  const confidence = Object.prototype.hasOwnProperty.call(point, confidenceKey)
    ? String(point[confidenceKey] || "unknown")
    : "unknown";
  return {
    source,
    status,
    confidence,
    explicit: hasExplicitSource || hasExplicitStatus || Object.prototype.hasOwnProperty.call(point, confidenceKey),
    label: sourceLabel(source),
    title: sourceTitle(source),
  };
}

function isUnavailable(point = {}, prefix) {
  const metadata = sourceMetadata(point, prefix);
  return metadata.source === "unavailable" || metadata.status.toLowerCase() === "unavailable";
}

function isExplicitUnavailable(point = {}, prefix) {
  const metadata = sourceMetadata(point, prefix);
  return metadata.explicit && (metadata.source === "unavailable" || metadata.status.toLowerCase() === "unavailable");
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatNumber(value, suffix = "", digits = 1) {
  const number = finiteNumber(value);
  if (number === null) {
    return "-";
  }
  const rounded = Number(number.toFixed(digits));
  return `${rounded}${suffix}`;
}

function formatMetric(value, suffix = "") {
  const number = finiteNumber(value);
  if (number === null) {
    return "-";
  }
  return `${number.toFixed(3)}${suffix}`;
}

function displayValue(point = {}, variableKey) {
  const variable = VARIABLE_DEFS[variableKey] || VARIABLE_DEFS.temperature;
  const prefix = variable.prefix;
  const unavailable = isUnavailable(point, prefix);
  if (unavailable && ["precip_probability", "wind", "cloud", "weather_code"].includes(variableKey)) {
    return "준비 중";
  }
  if (unavailable && ["temperature", "humidity"].includes(variableKey) && sourceMetadata(point, prefix).explicit) {
    return "준비 중";
  }

  if (variableKey === "temperature") {
    return formatNumber(point.temperature_c, "°C");
  }
  if (variableKey === "humidity") {
    return formatNumber(point.humidity_percent, "%");
  }
  if (variableKey === "precip_probability") {
    return formatNumber(point.precip_probability, "%", 0);
  }
  if (variableKey === "wind") {
    const speed = formatNumber(point.wind_speed_ms, "m/s");
    const direction = windDirectionLabel(point.wind_direction_deg);
    return speed === "-" ? "-" : `${speed} ${direction}`;
  }
  if (variableKey === "cloud") {
    const cover = finiteNumber(point.cloud_cover_percent);
    if (cover === null && point.cloud_label) {
      return String(point.cloud_label);
    }
    return formatNumber(cover, "%", 0);
  }
  if (variableKey === "weather_code") {
    return weatherLabel(point);
  }
  return "-";
}

function valueForVariable(point = {}, variableKey) {
  return displayValue(point, variableKey);
}

function numericValueForColor(point = {}, variableKey) {
  if (["temperature", "humidity"].includes(variableKey)) {
    const prefix = VARIABLE_DEFS[variableKey].prefix;
    const metadata = sourceMetadata(point, prefix);
    if (metadata.explicit && isUnavailable(point, prefix)) {
      return null;
    }
  }
  if (variableKey === "temperature") {
    return finiteNumber(point.temperature_c);
  }
  if (variableKey === "humidity") {
    return finiteNumber(point.humidity_percent);
  }
  if (variableKey === "precip_probability" && !isUnavailable(point, "precip_probability")) {
    return finiteNumber(point.precip_probability);
  }
  if (variableKey === "wind" && !isUnavailable(point, "wind")) {
    return finiteNumber(point.wind_speed_ms);
  }
  if (variableKey === "cloud" && !isUnavailable(point, "cloud")) {
    return finiteNumber(point.cloud_cover_percent);
  }
  return null;
}

function windDirectionLabel(degrees) {
  const number = finiteNumber(degrees);
  if (number === null) {
    return "";
  }
  const directions = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"];
  const index = Math.round((((number % 360) + 360) % 360) / 45) % 8;
  return `${Math.round(number)}° ${directions[index]}`;
}

function weatherLabel(point = {}) {
  const icon = point.weather_icon || "";
  const label = point.weather_label_ko || point.weather_label || point.weather_code || "-";
  return `${icon} ${label}`.trim();
}

function parseDate(value) {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function utcDateKey(date) {
  if (!date) {
    return "";
  }
  return date.toISOString().slice(0, 10);
}

function kstHour(date) {
  if (!date) {
    return null;
  }
  return (date.getUTCHours() + 9) % 24;
}

function dateKeyForPoint(point = {}) {
  if (typeof point.valid_date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(point.valid_date)) {
    return point.valid_date;
  }
  return utcDateKey(parseDate(point.valid_time || point.datetime || point.forecast_date || point.date || point.valid_date));
}

function explicitReferenceDate(run = {}) {
  const candidates = [
    run.forecast_init_time,
    run.forecast_init,
    run.issue_time,
    run.run_time,
    run.base_time,
    run.created_at,
  ].map(parseDate).filter(Boolean);
  if (candidates.length) {
    return candidates[0];
  }
  return null;
}

function referenceDate(run = {}, points = []) {
  const explicit = explicitReferenceDate(run);
  if (explicit) {
    return explicit;
  }
  const pointDates = points
    .map((point) => parseDate(point.valid_time || point.datetime || point.valid_date || point.forecast_date))
    .filter(Boolean)
    .sort((a, b) => a - b);
  return pointDates[0] || null;
}

function parseForecastDay(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = finiteNumber(value);
  if (number !== null) {
    return Math.round(number);
  }
  const match = String(value).trim().match(/D\s*\+\s*(\d+)/i);
  if (match) {
    return Number(match[1]);
  }
  return null;
}

function horizonDayForPoint(point = {}, run = {}, points = []) {
  for (const key of ["horizon_day", "forecast_day", "lead_day", "day_ahead"]) {
    const value = parseForecastDay(point[key]);
    if (value !== null) {
      return value;
    }
  }
  const valid = parseDate(point.valid_time || point.datetime || point.valid_date || point.forecast_date);
  const reference = explicitReferenceDate(run);
  if (valid && reference) {
    const validStart = Date.UTC(valid.getUTCFullYear(), valid.getUTCMonth(), valid.getUTCDate());
    const refStart = Date.UTC(reference.getUTCFullYear(), reference.getUTCMonth(), reference.getUTCDate());
    return Math.round((validStart - refStart) / 86400000);
  }
  return null;
}

function sortedUniqueDateKeys(points = []) {
  return [...new Set(points.map(dateKeyForPoint).filter(Boolean))].sort();
}

function choosePointForHorizon(points = [], horizonDay = 1, run = {}) {
  if (!points.length) {
    return null;
  }
  const matchedByDay = points.filter((point) => horizonDayForPoint(point, run, points) === horizonDay);
  if (matchedByDay.length) {
    return matchedByDay.slice().sort((left, right) => displayHourDistance(left, run) - displayHourDistance(right, run))[0] || null;
  }
  const hasExplicitHorizon = points.some((point) => horizonDayForPoint(point, run, points) !== null);
  if (hasExplicitHorizon) {
    return null;
  }
  const dateKeys = sortedUniqueDateKeys(points);
  const fallbackDate = dateKeys[horizonDay - 1];
  const matchedByDateOrder = fallbackDate ? points.filter((point) => dateKeyForPoint(point) === fallbackDate) : [];
  if (!matchedByDateOrder.length && dateKeys.length) {
    return null;
  }
  const candidates = matchedByDateOrder.length ? matchedByDateOrder : horizonDay === 1 ? points : [];
  return candidates.slice().sort((left, right) => displayHourDistance(left, run) - displayHourDistance(right, run))[0] || null;
}

function availableHorizonDays(points = [], run = {}) {
  if (!points.length) {
    return new Set();
  }
  const explicitDays = points.map((point) => horizonDayForPoint(point, run, points)).filter((day) => day !== null && day > 0);
  if (explicitDays.length) {
    return new Set(explicitDays);
  }
  return new Set(sortedUniqueDateKeys(points).slice(0, HORIZONS.length).map((_, index) => index + 1));
}

function hasHorizonData(points = [], horizonDay = 1, run = {}) {
  return availableHorizonDays(points, run).has(horizonDay);
}

function middayDistance(point = {}) {
  const date = parseDate(point.valid_time || point.datetime || point.valid_date || point.forecast_date);
  if (!date) {
    return 99;
  }
  return Math.abs(date.getUTCHours() - 12);
}

function referenceDisplayHourKst(run = {}) {
  const explicit = finiteNumber(run.map_reference_hour_kst ?? run.display_reference_hour_kst ?? run.reference_hour_kst);
  if (explicit !== null && run.lock_map_reference_hour_kst === true) {
    return ((Math.round(explicit) % 24) + 24) % 24;
  }
  const now = new Date();
  return kstHour(now);
}

function circularHourDistance(hour, referenceHour) {
  if (hour === null || referenceHour === null) {
    return 99;
  }
  const diff = Math.abs(hour - referenceHour);
  return Math.min(diff, 24 - diff);
}

function displayHourDistance(point = {}, run = {}) {
  const date = parseDate(point.valid_time || point.datetime || point.forecast_date || point.date);
  if (!date) {
    return middayDistance(point);
  }
  return circularHourDistance(kstHour(date), referenceDisplayHourKst(run));
}

function stationId(point = {}) {
  return String(point.station_id || point.id || "");
}

function stationName(point = {}) {
  return String(point.station_name || point.name || stationId(point) || "지역");
}

function regionName(point = {}) {
  return String(point.region_class || point.region || point.province || "전국");
}

function stationDisplayPriority(point = {}) {
  const id = stationId(point);
  const name = stationName(point);
  if (PRIMARY_STATION_IDS.has(id) || PRIMARY_STATION_NAMES.has(name)) {
    return 1;
  }
  if (SECONDARY_STATION_IDS.has(id)) {
    return 2;
  }
  return 3;
}

function visibleEntriesForZoom(entries = [], zoom = 7, selectedStationId = "") {
  if (!entries.length) {
    return [];
  }
  const threshold = zoom < 8 ? 1 : zoom < 9 ? 2 : 3;
  const visible = entries.filter((entry) => stationDisplayPriority(entry.point) <= threshold);
  if (selectedStationId && !visible.some((entry) => entry.id === selectedStationId)) {
    const selected = entries.find((entry) => entry.id === selectedStationId);
    if (selected) {
      visible.push(selected);
    }
  }
  return visible.length ? visible : entries.slice(0, Math.min(entries.length, 14));
}

function visiblePointsForZoom(points = [], zoom = 7) {
  return points.filter((point) => stationDisplayPriority(point) <= (zoom < 8 ? 1 : zoom < 9 ? 2 : 3));
}

function groupPointsByStation(points = []) {
  const groups = new Map();
  points.forEach((point) => {
    const id = stationId(point);
    if (!id) {
      return;
    }
    if (!groups.has(id)) {
      groups.set(id, []);
    }
    groups.get(id).push(point);
  });
  return groups;
}

function selectedPointsByStation(points = [], horizonDay = 1, run = {}) {
  const groups = groupPointsByStation(points);
  return [...groups.entries()].map(([id, stationPoints], index) => {
    const point = choosePointForHorizon(stationPoints, horizonDay, run);
    return { id, point, stationPoints, index };
  }).filter((entry) => entry.point);
}

function pointsForHorizon(points = [], horizonDay = 1, run = {}) {
  return selectedPointsByStation(points, horizonDay, run).map((entry) => entry.point);
}

function variableHasDisplayData(point = {}, variableKey) {
  const value = displayValue(point, variableKey);
  return value !== "-" && value !== "준비 중";
}

function variableReadiness(variableKey, points = [], horizonDay = 1, run = {}) {
  const horizonPoints = pointsForHorizon(points, horizonDay, run);
  if (!horizonPoints.length) {
    return { label: "데이터 없음", ready: false, source: "unavailable" };
  }
  const variable = VARIABLE_DEFS[variableKey] || VARIABLE_DEFS.temperature;
  const explicit = horizonPoints.map((point) => sourceMetadata(point, variable.prefix)).find((metadata) => metadata.explicit);
  const anyDisplayData = horizonPoints.some((point) => variableHasDisplayData(point, variableKey));
  if (explicit && explicit.source !== "unavailable" && explicit.status.toLowerCase() !== "unavailable") {
    return { label: explicit.label, ready: anyDisplayData, source: explicit.source };
  }
  return { label: anyDisplayData ? "source 미상" : "준비 중", ready: anyDisplayData, source: explicit?.source || "unavailable" };
}

function chooseDefaultStationId(points = [], run = {}, horizonDay = 1) {
  const entries = selectedPointsByStation(points, horizonDay, run);
  if (!entries.length) {
    return "";
  }
  const seoulByName = entries.find((entry) => stationName(entry.point).includes("서울"));
  if (seoulByName) {
    return seoulByName.id;
  }
  const seoulById = entries.find((entry) => entry.id === "108");
  return (seoulById || entries[0]).id;
}

function ensureSelectedHorizon() {
  const available = availableHorizonDays(state.allPoints, state.latest.run || {});
  if (!available.size) {
    state.selectedHorizon = 1;
    return;
  }
  if (!available.has(state.selectedHorizon)) {
    const preferred = HORIZONS.find((horizon) => available.has(horizon.day));
    state.selectedHorizon = preferred ? preferred.day : [...available][0];
  }
}

function coordinateForPoint(point = {}, index = 0) {
  const lat = finiteNumber(point.lat || point.latitude);
  const lon = finiteNumber(point.lon || point.longitude);
  if (lat !== null && lon !== null) {
    const minLon = 124.5;
    const maxLon = 131.5;
    const minLat = 33.0;
    const maxLat = 38.8;
    const x = Math.min(86, Math.max(14, ((lon - minLon) / (maxLon - minLon)) * 100));
    const y = Math.min(92, Math.max(8, ((maxLat - lat) / (maxLat - minLat)) * 100));
    return { x, y };
  }
  const fallback = FALLBACK_COORDINATES[stationId(point)] || FALLBACK_COORDINATES[stationName(point)] || REGION_COORDINATES[regionName(point)];
  if (fallback) {
    return { x: fallback.x, y: fallback.y };
  }
  const columns = 4;
  return { x: 28 + (index % columns) * 14, y: 24 + Math.floor(index / columns) * 12 };
}

function latLonForPoint(point = {}, index = 0) {
  const lat = finiteNumber(point.lat || point.latitude);
  const lon = finiteNumber(point.lon || point.longitude);
  if (lat !== null && lon !== null) {
    return [lat, lon];
  }
  const fallback = FALLBACK_LATLON[stationId(point)] || FALLBACK_LATLON[stationName(point)] || REGION_LATLON[regionName(point)];
  if (fallback) {
    return [fallback.lat, fallback.lon];
  }
  const coordinate = coordinateForPoint(point, index);
  const lonFromX = 124.5 + ((coordinate.x - 14) / 72) * (131.5 - 124.5);
  const latFromY = 38.8 - ((coordinate.y - 8) / 84) * (38.8 - 33.0);
  return [latFromY, lonFromX];
}

function leafletAvailable() {
  return typeof window !== "undefined"
    && typeof document !== "undefined"
    && typeof window.L !== "undefined"
    && Boolean(document.getElementById("leafletMap"));
}

function initializeLeafletMap() {
  if (!leafletAvailable()) {
    return null;
  }
  const container = document.getElementById("forecastMap");
  if (container) {
    container.classList.add("leaflet-active");
  }
  if (leafletState.map) {
    return leafletState.map;
  }
  const map = window.L.map("leafletMap", {
    center: [36.2, 127.8],
    zoom: 7,
    minZoom: 6,
    maxZoom: 11,
    maxBounds: KOREA_MAP_BOUNDS,
    maxBoundsViscosity: 0.7,
    zoomControl: true,
    attributionControl: true,
  });
  window.L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors",
  }).addTo(map);
  map.on("zoomend", () => {
    if (leafletState.updatingBounds) {
      return;
    }
    renderLeafletMarkers(selectedPointsByStation(state.allPoints, state.selectedHorizon, state.latest.run || {}), { fit: false });
  });
  map.fitBounds(KOREA_MAP_BOUNDS, { padding: [18, 18] });
  setTimeout(() => map.invalidateSize(), 0);
  leafletState.map = map;
  return map;
}

function clearLeafletMarkers() {
  if (!leafletState.map) {
    return;
  }
  leafletState.markers.forEach((marker) => marker.remove());
  leafletState.markers = [];
}

function leafletMarkerHtml(point = {}, value = "", color = "#98a2b3", selected = false, empty = false) {
  const classes = ["leaflet-forecast-marker"];
  if (selected) {
    classes.push("selected");
  }
  if (empty) {
    classes.push("empty-anchor");
  }
  return [
    `<div class="${classes.join(" ")}" style="--marker-color:${escapeHtml(color)}">`,
    `<span class="marker-name">${escapeHtml(stationName(point))}</span>`,
    `<strong class="marker-value">${escapeHtml(value)}</strong>`,
    "</div>",
  ].join("");
}

function addLeafletMarker(point = {}, value = "", options = {}) {
  const map = leafletState.map;
  if (!map || typeof window === "undefined" || typeof window.L === "undefined") {
    return null;
  }
  const color = options.color || "#98a2b3";
  const marker = window.L.marker(latLonForPoint(point, options.index || 0), {
    icon: window.L.divIcon({
      className: "forecast-div-icon",
      html: leafletMarkerHtml(point, value, color, Boolean(options.selected), Boolean(options.empty)),
      iconSize: [112, 56],
      iconAnchor: [56, 56],
    }),
    keyboard: !options.empty,
    interactive: !options.empty,
  });
  if (!options.empty && options.onClick) {
    marker.on("click", options.onClick);
  }
  marker.addTo(map);
  leafletState.markers.push(marker);
  return marker;
}

function renderLeafletMarkers(entries = [], options = {}) {
  const map = initializeLeafletMap();
  if (!map) {
    return false;
  }
  clearLeafletMarkers();
  const zoom = map.getZoom ? map.getZoom() : 7;
  const visibleEntries = visibleEntriesForZoom(entries, zoom, state.selectedStationId);
  const emptyAnchors = visiblePointsForZoom(EMPTY_STATION_ANCHORS, zoom);
  const markerPoints = visibleEntries.length
    ? visibleEntries.map((entry) => entry.point)
    : emptyAnchors;
  if (!entries.length) {
    emptyAnchors.forEach((point, index) => addLeafletMarker(point, "데이터 없음", {
      color: "#98a2b3",
      empty: true,
      index,
    }));
  } else {
    visibleEntries.forEach((entry, index) => {
      const { point } = entry;
      addLeafletMarker(point, valueForVariable(point, state.selectedVariable), {
        color: colorForValue(point, state.selectedVariable),
        selected: entry.id === state.selectedStationId,
        index,
        onClick: async () => {
          state.selectedStationId = entry.id;
          await loadSelectedStationData();
          await renderAll(false);
        },
      });
    });
  }
  const bounds = markerPoints.map((point, index) => latLonForPoint(point, index));
  if (options.fit !== false && !leafletState.initialFitDone) {
    leafletState.updatingBounds = true;
    if (bounds.length) {
      map.fitBounds(bounds, { padding: [46, 46], maxZoom: 7 });
    } else {
      map.fitBounds(KOREA_MAP_BOUNDS, { padding: [18, 18] });
    }
    leafletState.initialFitDone = true;
    setTimeout(() => {
      leafletState.updatingBounds = false;
    }, 0);
  }
  return true;
}

function colorForValue(point = {}, variableKey = "temperature") {
  if (isUnavailable(point, VARIABLE_DEFS[variableKey]?.prefix || variableKey) && ["precip_probability", "wind", "cloud", "weather_code"].includes(variableKey)) {
    return "#98a2b3";
  }
  if (variableKey === "weather_code") {
    const label = weatherLabel(point);
    if (label.includes("비") || label.toLowerCase().includes("rain")) {
      return "#2563eb";
    }
    if (label.includes("흐") || label.toLowerCase().includes("cloud")) {
      return "#64748b";
    }
    if (label.includes("구름")) {
      return "#38bdf8";
    }
    return "#f59e0b";
  }
  const value = numericValueForColor(point, variableKey);
  if (value === null) {
    return "#98a2b3";
  }
  if (variableKey === "temperature") {
    if (value >= 28) return "#ef4444";
    if (value >= 22) return "#f97316";
    if (value >= 15) return "#eab308";
    if (value >= 5) return "#22c55e";
    return "#3b82f6";
  }
  if (variableKey === "humidity" || variableKey === "precip_probability" || variableKey === "cloud") {
    if (value >= 80) return "#1d4ed8";
    if (value >= 60) return "#0ea5e9";
    if (value >= 40) return "#14b8a6";
    if (value >= 20) return "#a3e635";
    return "#facc15";
  }
  if (variableKey === "wind") {
    if (value >= 10) return "#7c3aed";
    if (value >= 6) return "#2563eb";
    if (value >= 3) return "#0ea5e9";
    return "#22c55e";
  }
  return "#3b82f6";
}

function renderBadge(parent, className, text, title = "") {
  const badge = createElement("span", className, text);
  if (title) {
    badge.title = title;
  }
  parent.appendChild(badge);
  return badge;
}

function SourceBadge(parent, label, point = {}, prefix) {
  const metadata = sourceMetadata(point, prefix);
  const className = `source-badge source-${metadata.source.replaceAll("_", "-")}`;
  return renderBadge(parent, className, `${label}: ${metadata.label}`, `${metadata.title} · status=${metadata.status} · confidence=${metadata.confidence}`);
}

function ConfidenceBadge(parent, point = {}, prefix = "") {
  const metadata = prefix ? sourceMetadata(point, prefix) : { confidence: point.confidence || "unknown" };
  return renderBadge(parent, "confidence-badge", `신뢰도: ${metadata.confidence || point.confidence || "unknown"}`);
}

function sourceStatusConfidence(point = {}, prefix) {
  const metadata = sourceMetadata(point, prefix);
  return `${metadata.label} · status=${metadata.status} · confidence=${metadata.confidence}`;
}

function CaveatBanner(status = {}) {
  const banner = document.getElementById("caveatBanner");
  if (!banner) {
    return;
  }
  banner.textContent = "본 사이트는 AI 기반 기상 예측 모델의 미래 예측값을 지도 형태로 제공하는 연구/베타 서비스입니다. 실제 기상청 예보와 다를 수 있습니다.";
  if (status.operational_valid === true && status.site_readiness === "PASS") {
    banner.classList.add("caveat-banner-soft");
  } else {
    banner.classList.remove("caveat-banner-soft");
  }
}

function DateSelector() {
  const container = document.getElementById("dateSelector");
  if (!container) {
    return;
  }
  const available = availableHorizonDays(state.allPoints, state.latest.run || {});
  container.replaceChildren();
  HORIZONS.forEach((horizon) => {
    const hasData = available.has(horizon.day);
    const button = createElement("button", "tab-button");
    button.type = "button";
    button.dataset.horizon = String(horizon.day);
    if (!hasData) {
      button.disabled = true;
      button.title = "선택한 날짜의 예측 데이터가 없습니다.";
    }
    appendTextElement(button, "span", horizon.label, "tab-label");
    appendTextElement(button, "span", hasData ? "데이터 있음" : "데이터 없음", "tab-sub");
    if (horizon.day === state.selectedHorizon) {
      button.classList.add("active");
      button.setAttribute("aria-current", "true");
    }
    button.addEventListener("click", async () => {
      if (button.disabled) {
        return;
      }
      state.selectedHorizon = horizon.day;
      await renderAll();
    });
    container.appendChild(button);
  });
}

function VariableSelector() {
  const container = document.getElementById("variableSelector");
  if (!container) {
    return;
  }
  container.replaceChildren();
  Object.values(VARIABLE_DEFS).forEach((variable) => {
    const readiness = variableReadiness(variable.key, state.allPoints, state.selectedHorizon, state.latest.run || {});
    const button = createElement("button", `metric-button ${readiness.ready ? "" : "variable-unavailable"}`.trim());
    button.type = "button";
    button.dataset.variable = variable.key;
    button.title = `${variable.label}: ${readiness.label}`;
    appendTextElement(button, "span", variable.label, "metric-label");
    appendTextElement(button, "span", readiness.label, "metric-source");
    if (variable.key === state.selectedVariable) {
      button.classList.add("active");
      button.setAttribute("aria-current", "true");
    }
    button.addEventListener("click", async () => {
      state.selectedVariable = variable.key;
      await renderAll();
    });
    container.appendChild(button);
  });
}

function ForecastMap() {
  const layer = document.getElementById("markerLayer");
  if (!layer) {
    return;
  }
  const emptyState = document.getElementById("mapEmptyState");
  layer.replaceChildren();
  const entries = selectedPointsByStation(state.allPoints, state.selectedHorizon, state.latest.run || {});
  if (emptyState) {
    if (!state.allPoints.length) {
      emptyState.hidden = false;
      emptyState.textContent = "예측 데이터가 없습니다. forecast exporter를 실행해 latest forecast_points.json을 생성하세요.";
    } else if (!entries.length) {
      emptyState.hidden = false;
      emptyState.textContent = "선택한 날짜의 예측 데이터가 없습니다.";
    } else {
      emptyState.hidden = true;
      emptyState.textContent = "";
    }
  }
  if (renderLeafletMarkers(entries)) {
    return;
  }
  if (!entries.length) {
    visiblePointsForZoom(EMPTY_STATION_ANCHORS, 7).forEach((point, index) => {
      const coordinate = coordinateForPoint(point, index);
      const marker = createElement("button", "map-marker empty-anchor");
      marker.type = "button";
      marker.disabled = true;
      marker.style.left = `${coordinate.x}%`;
      marker.style.top = `${coordinate.y}%`;
      marker.style.setProperty("--marker-color", "#98a2b3");
      appendTextElement(marker, "span", stationName(point), "marker-name");
      appendTextElement(marker, "strong", "데이터 없음", "marker-value");
      layer.appendChild(marker);
    });
    return;
  }
  if (!state.selectedStationId && entries.length) {
    state.selectedStationId = entries[0].id;
  }
  visibleEntriesForZoom(entries, 7, state.selectedStationId).forEach((entry, index) => {
    const { point } = entry;
    const coordinate = coordinateForPoint(point, index);
    const button = createElement("button", "map-marker");
    button.type = "button";
    button.style.left = `${coordinate.x}%`;
    button.style.top = `${coordinate.y}%`;
    button.style.setProperty("--marker-color", colorForValue(point, state.selectedVariable));
    button.dataset.stationId = entry.id;
    if (entry.id === state.selectedStationId) {
      button.classList.add("selected");
    }
    appendTextElement(button, "span", stationName(point), "marker-name");
    appendTextElement(button, "strong", valueForVariable(point, state.selectedVariable), "marker-value");
    button.addEventListener("click", async () => {
      state.selectedStationId = entry.id;
      await loadSelectedStationData();
      await renderAll(false);
    });
    layer.appendChild(button);
  });
}

function renderLegend() {
  const legend = document.getElementById("mapLegend");
  if (!legend) {
    return;
  }
  const variable = VARIABLE_DEFS[state.selectedVariable] || VARIABLE_DEFS.temperature;
  legend.replaceChildren();
  appendTextElement(legend, "h2", variable.legend, "legend-title");
  const body = createElement("div", "legend-body");
  body.appendChild(createElement("div", `legend-bar ${variable.className}`));
  const ticks = createElement("div", "legend-ticks");
  variable.ticks.forEach((tick) => appendTextElement(ticks, "span", tick));
  body.appendChild(ticks);
  legend.appendChild(body);
}

function selectedPoint() {
  const stationPoints = state.selectedStationPoints.length
    ? state.selectedStationPoints
    : state.allPoints.filter((point) => stationId(point) === state.selectedStationId);
  return choosePointForHorizon(stationPoints, state.selectedHorizon, state.latest.run || {}) || stationPoints[0] || null;
}

function renderSourceBadges(parent, point = {}) {
  const badges = createElement("div", "source-badges");
  SourceBadge(badges, "기온", point, "temperature");
  SourceBadge(badges, "습도", point, "humidity");
  SourceBadge(badges, "강수확률", point, "precip_probability");
  SourceBadge(badges, "바람", point, "wind");
  SourceBadge(badges, "구름", point, "cloud");
  SourceBadge(badges, "날씨상태", point, "weather_code");
  parent.appendChild(badges);
}

function appendMetricRow(parent, label, value, sourcePoint, prefix) {
  const row = createElement("div", "metric-row");
  appendTextElement(row, "span", label);
  appendTextElement(row, "strong", value);
  if (prefix) {
    const metadata = sourceMetadata(sourcePoint, prefix);
    const badge = createElement("em", `mini-source source-${metadata.source.replaceAll("_", "-")}`, metadata.label);
    badge.title = `${metadata.title} · status=${metadata.status} · confidence=${metadata.confidence}`;
    row.appendChild(badge);
  }
  parent.appendChild(row);
}

function RegionForecastPanel() {
  const panel = document.getElementById("selectedRegionPanel");
  if (!panel) {
    return;
  }
  panel.replaceChildren();
  const point = selectedPoint();
  if (!point) {
    appendTextElement(panel, "h2", "선택 지역");
    appendTextElement(panel, "p", emptyForecastMessage(), "empty-message");
    return;
  }
  const horizon = HORIZONS.find((entry) => entry.day === state.selectedHorizon) || HORIZONS[0];
  appendTextElement(panel, "p", `${regionName(point)} · ${horizon.shortLabel}`, "eyebrow");
  appendTextElement(panel, "h2", stationName(point));
  appendTextElement(panel, "p", displayValue(point, "weather_code"), "weather-line");

  const metrics = createElement("div", "metric-list");
  appendMetricRow(metrics, "선택 변수", valueForVariable(point, state.selectedVariable), point, VARIABLE_DEFS[state.selectedVariable].prefix);
  appendMetricRow(metrics, "평균 기온", displayValue(point, "temperature"), point, "temperature");
  appendMetricRow(metrics, "최고/최저 기온", highLowText(point), point, "temperature");
  appendMetricRow(metrics, "습도", displayValue(point, "humidity"), point, "humidity");
  appendMetricRow(metrics, "강수확률", displayValue(point, "precip_probability"), point, "precip_probability");
  appendMetricRow(metrics, "바람", displayValue(point, "wind"), point, "wind");
  appendMetricRow(metrics, "구름", displayValue(point, "cloud"), point, "cloud");
  appendMetricRow(metrics, "날씨상태", displayValue(point, "weather_code"), point, "weather_code");
  panel.appendChild(metrics);

  const badgeRow = createElement("div", "source-badges source-badges-panel");
  ConfidenceBadge(badgeRow, point, VARIABLE_DEFS[state.selectedVariable].prefix);
  panel.appendChild(badgeRow);
  renderSourceBadges(panel, point);
}

function highLowText(point = {}) {
  if (isExplicitUnavailable(point, "temperature")) {
    return "준비 중";
  }
  const maxValue = point.temp_max_c ?? point.temperature_max_c ?? point.high_temperature_c;
  const minValue = point.temp_min_c ?? point.temperature_min_c ?? point.low_temperature_c;
  const maxText = formatNumber(maxValue, "°C");
  const minText = formatNumber(minValue, "°C");
  if (maxText === "-" && minText === "-") {
    return displayValue(point, "temperature");
  }
  return `${maxText} / ${minText}`;
}

function deriveDailyRows(points = []) {
  const rows = [];
  HORIZONS.forEach((horizon) => {
    const point = choosePointForHorizon(points, horizon.day, state.latest.run || {});
    if (point) {
      rows.push({ day: horizon.day, point });
    }
  });
  return rows;
}

function dailySummaryForDay(day, point = {}) {
  const pointDate = dateKeyForPoint(point);
  const match = state.selectedDaily.find((entry) => String(entry.date) === pointDate) || {};
  return {
    tempMax: match.temp_max_c ?? point.temp_max_c ?? point.temperature_max_c,
    tempMin: match.temp_min_c ?? point.temp_min_c ?? point.temperature_min_c,
    humidity: match.humidity_mean_percent ?? point.humidity_percent,
  };
}

function ForecastTable() {
  const tbody = document.querySelector("#forecastTable tbody");
  if (!tbody) {
    return;
  }
  tbody.replaceChildren();
  const stationPoints = state.selectedStationPoints.length
    ? state.selectedStationPoints
    : state.allPoints.filter((point) => stationId(point) === state.selectedStationId);
  const rows = deriveDailyRows(stationPoints);
  if (!rows.length) {
    const row = createElement("tr");
    const cell = createElement("td", "empty-cell", state.allPoints.length ? "선택한 날짜의 예측 데이터가 없습니다." : "예측 데이터가 없습니다. latest forecast_points.json을 생성했는지 확인하세요.");
    cell.colSpan = 8;
    row.appendChild(cell);
    tbody.appendChild(row);
    return;
  }
  rows.forEach(({ day, point }) => {
    const daily = dailySummaryForDay(day, point);
    const temperatureUnavailable = isExplicitUnavailable(point, "temperature");
    const row = createElement("tr");
    const values = [
      `${dateKeyForPoint(point) || `D+${day}`} · D+${day}`,
      displayValue(point, "weather_code"),
      temperatureUnavailable ? "준비 중" : formatNumber(daily.tempMax, "°C"),
      temperatureUnavailable ? "준비 중" : formatNumber(daily.tempMin, "°C"),
      isUnavailable(point, "humidity") && sourceMetadata(point, "humidity").explicit ? "준비 중" : formatNumber(daily.humidity, "%"),
      displayValue(point, "precip_probability"),
      displayValue(point, "wind"),
      sourceMetadata(point, "weather_code").confidence || point.confidence || "unknown",
    ];
    values.forEach((value) => appendTextElement(row, "td", value));
    tbody.appendChild(row);
  });
}

function emptyForecastMessage() {
  if (!state.allPoints.length) {
    return "예측 데이터가 없습니다. data/forecasts/latest/forecast_points.json을 생성했는지 확인하세요.";
  }
  return "선택한 날짜/변수에 해당하는 예측값이 없습니다.";
}

function ModelPerformanceSummary() {
  const panel = document.getElementById("modelPerformanceSummary");
  if (!panel) {
    return;
  }
  const status = state.modelStatus || {};
  panel.replaceChildren();
  appendTextElement(panel, "h2", "모델 성능 요약");
  if (!hasModelPerformance(status)) {
    appendTextElement(panel, "p", "모델 성능 정보 없음", "empty-message");
    appendTextElement(panel, "p", "forecast_run.json, production_model_manifest.json 또는 /api/model/status 응답에 RMSE/site readiness 정보를 연결하세요.", "note-line");
    return;
  }
  const statusText = status.operational_valid === true ? "Operational Beta" : "Research Beta / WARN";
  const summary = createElement("div", "model-grid");
  appendMetricRow(summary, "기온 RMSE", formatMetric(status.temperature_rmse ?? status.temp_rmse, "°C"), {}, "");
  appendMetricRow(summary, "습도 RMSE", formatMetric(status.humidity_rmse, "%p"), {}, "");
  appendMetricRow(summary, "벤치마크 신뢰도", status.benchmark_reliability || "unknown", {}, "");
  appendMetricRow(summary, "모델 상태", statusText, {}, "");
  panel.appendChild(summary);

  const badges = createElement("div", "status-badges");
  renderBadge(badges, `status-badge ${status.operational_valid === true ? "status-pass" : "status-warn"}`, status.operational_valid === true ? "Operational valid" : "Research Beta");
  renderBadge(badges, `status-badge status-${String(status.site_readiness || "WARN").toLowerCase()}`, `Site ${status.site_readiness || "WARN"}`);
  renderBadge(badges, `status-badge status-${String(status.v4_c_gate_status || "FAIL").toLowerCase()}`, `V4-C ${status.v4_c_gate_status || "FAIL"}`);
  if (status.model_improvement_frozen) {
    renderBadge(badges, "status-badge status-frozen", "model freeze");
  }
  panel.appendChild(badges);

  appendTextElement(panel, "p", "본 예측은 AI 모델 기반 정보이며 실제 기상청 예보와 다를 수 있습니다.", "note-line");
  (status.warnings || []).slice(0, 4).forEach((warning) => appendTextElement(panel, "p", warning, "note-line"));
}

function hasModelPerformance(status = {}) {
  return (
    finiteNumber(status.temperature_rmse ?? status.temp_rmse) !== null
    || finiteNumber(status.humidity_rmse) !== null
    || Boolean(status.benchmark_reliability)
    || Boolean(status.site_readiness)
    || Boolean(status.temperature_status)
    || Boolean(status.humidity_status)
    || Boolean(status.model_version)
  );
}

function issuedAtText(latest = {}, status = {}) {
  const run = latest.run || {};
  const raw = run.forecast_init_time || run.forecast_init || run.issue_time || run.run_time || run.created_at || status.latest_forecast_run_time;
  const parsed = parseDate(raw);
  if (!parsed) {
    return "데이터 없음";
  }
  const kst = new Date(parsed.getTime() + 9 * 60 * 60 * 1000);
  const yyyy = kst.getUTCFullYear();
  const mm = String(kst.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(kst.getUTCDate()).padStart(2, "0");
  const hh = String(kst.getUTCHours()).padStart(2, "0");
  const mi = String(kst.getUTCMinutes()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${mi} KST`;
}

function updateHeader() {
  const issued = document.getElementById("issuedAtText");
  if (issued) {
    issued.textContent = issuedAtText(state.latest, state.modelStatus);
  }
  const pill = document.getElementById("modelStatusPill");
  if (pill) {
    const status = state.modelStatus || {};
    pill.textContent = `${status.operational_valid === true ? "Operational Beta" : "Research Beta"} · Site ${status.site_readiness || "WARN"}`;
  }
}

async function loadSelectedStationData() {
  if (!state.selectedStationId) {
    state.selectedStationPoints = [];
    state.selectedDaily = [];
    state.stationDataWarning = "";
    return;
  }
  if (state.stationCache.has(state.selectedStationId)) {
    const cached = state.stationCache.get(state.selectedStationId);
    state.selectedStationPoints = cached.points;
    state.selectedDaily = cached.daily;
    state.stationDataWarning = "";
    return;
  }
  const encodedStationId = encodeURIComponent(state.selectedStationId);
  state.stationDataWarning = "";
  try {
    const [stationPayload, hourlyPayload, dailyPayload] = await Promise.all([
      fetchJson(`/api/forecast/station/${encodedStationId}`),
      fetchJson(`/api/forecast/station/${encodedStationId}/hourly`),
      fetchJson(`/api/forecast/station/${encodedStationId}/daily`),
    ]);
    const stationPoints = Array.isArray(hourlyPayload) && hourlyPayload.length
      ? hourlyPayload
      : Array.isArray(stationPayload.points)
        ? stationPayload.points
        : [];
    const daily = Array.isArray(dailyPayload?.daily) ? dailyPayload.daily : [];
    state.stationCache.set(state.selectedStationId, { points: stationPoints, daily });
    state.selectedStationPoints = stationPoints;
    state.selectedDaily = daily;
    state.stationDataWarning = "";
  } catch (error) {
    const fallbackPoints = state.allPoints.filter((point) => stationId(point) === state.selectedStationId);
    state.selectedStationPoints = fallbackPoints;
    state.selectedDaily = [];
    state.stationDataWarning = `지역 상세 API를 불러오지 못해 latest forecast point로 제한 표시합니다: ${error.message || "unknown"}`;
  }
}

function StationWarning() {
  const card = document.getElementById("stationWarning");
  const text = document.getElementById("stationWarningText");
  if (!card || !text) {
    return;
  }
  if (!state.stationDataWarning) {
    card.hidden = true;
    text.textContent = "";
    return;
  }
  card.hidden = false;
  text.textContent = state.stationDataWarning;
}

async function renderAll(shouldLoadStation = true) {
  ensureSelectedHorizon();
  DateSelector();
  VariableSelector();
  updateHeader();
  CaveatBanner(state.modelStatus);
  if (shouldLoadStation) {
    await loadSelectedStationData();
  }
  ForecastMap();
  renderLegend();
  StationWarning();
  RegionForecastPanel();
  ForecastTable();
  ModelPerformanceSummary();
}

function showError(message) {
  const panel = document.getElementById("selectedRegionPanel");
  if (!panel) {
    return;
  }
  panel.replaceChildren();
  appendTextElement(panel, "h2", "데이터 준비 중");
  appendTextElement(panel, "p", message);
}

async function main() {
  try {
    const latest = await fetchLatestForecast();
    const modelStatus = await fetchModelStatus();
    state.latest = normalizeLatestPayload(latest || {});
    state.modelStatus = modelStatus || {};
    state.allPoints = state.latest.points;
    ensureSelectedHorizon();
    state.selectedStationId = chooseDefaultStationId(state.allPoints, state.latest.run || {}, state.selectedHorizon);
    await renderAll(true);
  } catch (error) {
    state.latest = { run: {}, points: [] };
    state.modelStatus = {};
    state.allPoints = [];
    state.selectedStationId = "";
    DateSelector();
    VariableSelector();
    ForecastMap();
    renderLegend();
    ModelPerformanceSummary();
    CaveatBanner({});
    showError("Forecast API 또는 latest forecast artifact를 먼저 생성하세요.");
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  main();
}

if (typeof module !== "undefined") {
  module.exports = {
    state,
    HORIZONS,
    VARIABLE_DEFS,
    SOURCE_LABELS,
    SOURCE_FIELD_NAMES,
    EMPTY_STATION_ANCHORS,
    normalizeLatestPayload,
    embeddedForecastData,
    embeddedModelStatusData,
    fetchFirstJson,
    fetchLatestForecast,
    fetchModelStatus,
    sourceLabel,
    sourceMetadata,
    sourceStatusConfidence,
    isUnavailable,
    isExplicitUnavailable,
    displayValue,
    valueForVariable,
    dateKeyForPoint,
    choosePointForHorizon,
    horizonDayForPoint,
    availableHorizonDays,
    hasHorizonData,
    selectedPointsByStation,
    pointsForHorizon,
    variableReadiness,
    chooseDefaultStationId,
    stationDisplayPriority,
    visibleEntriesForZoom,
    visiblePointsForZoom,
    coordinateForPoint,
    latLonForPoint,
    colorForValue,
    leafletAvailable,
    leafletMarkerHtml,
    DateSelector,
    VariableSelector,
    ForecastMap,
    ForecastTable,
    weatherLabel,
    windDirectionLabel,
    highLowText,
    kstHour,
    referenceDisplayHourKst,
    displayHourDistance,
    deriveDailyRows,
    hasModelPerformance,
  };
}
