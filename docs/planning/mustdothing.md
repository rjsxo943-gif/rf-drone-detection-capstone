전체적인 진행순서
    1. 부팅 후 바로 자동 calibration 하지 않음
    2. 사용자가 noise calibration / phase-gain calibration을 각각 trigger
    3. scan mode는 time-domain power로 가볍게 후보 탐색
    4. 후보만 STFT + CNN
    5. CNN이 애매하면 3~5 blocks 추가 확인
    6. Drone-like 확률이 충분할 때만 AoA 계산
    7. 대역 유지 중 coherence가 낮으면 바로 버리지 않고 일정 block 더 확인
    8. 조건이 계속 나쁘면 scan mode로 복귀

phase offset 선형근사식 정의

cnn model 학습

코드 무한 루프 자동화 구성
    run_system.py
        ↓
    SystemStateMachine
            ↓
    IDLE
    ├─ n 입력 → NOISE_CALIBRATION
    ├─ p 입력 → PHASE_GAIN_CALIBRATION
        Calibration 기능 1차 구현 완료.
        이제 runtime/state_machine.py에서 메뉴형 calibration mode로 묶으면 된다.
    ├─ s 입력 → SCAN
    └─ q 입력 → 종료

    SCAN
    ├─ energy 미통과 → 다음 주파수
    ├─ energy 통과 + CNN 확실 → BAND_HOLD
    ├─ energy 통과 + CNN 애매 → CANDIDATE_CONFIRM
    └─ CNN 불만족 → SCAN 유지

    CANDIDATE_CONFIRM
    ├─ 추가 block에서 조건 만족 → BAND_HOLD
    └─ 조건 불만족 → SCAN
    이건 안만들듯

    BAND_HOLD
    ├─ CNN drone 확률 높고 coherence 높음 → AoA 계산, 유지
    ├─ coherence 낮음 → lost_count 증가
    └─ lost_count 초과 → SCAN 복귀


    IDLE
    → 사용자가 명령 입력

    NOISE_CALIBRATION
    → 신호원 없이 50 blocks
    → noise floor / threshold 저장

    PHASE_GAIN_CALIBRATION
    → 신호원 정면 0도
    → 50 blocks
    → gain correction / phase offset 저장

    1. DC offset
   → 매 block 평균 제거

    2. noise calibration
    → 실행 환경마다 수행 권장

    3. phase/gain calibration
    → 매번 신호원 놓고 하지 않음
    → 사전에 주파수별 측정
    → phase_gain_model.json으로 저장
    → scan 중 center_freq에 맞춰 자동 적용

    4. p 모드
    → 일상 운용용이 아니라 검증/재측정/모델 갱신용으로 유지

    SCAN
    → 주파수 sweep
    → 2~3 blocks time power 검사
    → threshold 통과하면 STFT + CNN
    → CNN 확실하면 BAND_HOLD
    → CNN 애매하면 같은 대역에서 3~5 blocks 추가 확인
    → 아니면 다음 대역

    BAND_HOLD
    → 같은 대역 유지
    → CNN drone 확률 + coherence 확인
    → 조건 좋으면 AoA 계산
    → 조건 나쁘면 lost_count 증가
    → lost_count 초과하면 SCAN 복귀

    
대쉬보드 만들기

