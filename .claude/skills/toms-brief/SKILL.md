---
name: toms-brief
description: 탐스영어학원 관리자용 '오늘의 브리핑'을 만든다. 구글 드라이브의 운영시트를 받아 규칙 엔진을 돌리고 결과를 드라이브에 올린다. 아침 브리핑, 오늘 할 일, 관리자 브리핑 요청에 사용.
---

# 오늘의 브리핑

관리자가 출근해서 여는 화면을 만든다. 원장은 이 과정에 관여하지 않는다.

## 순서

1. **운영시트 받기.** 구글 드라이브에서 제목에 `탐스 운영시트`가 든 파일을 찾는다.
   `download_file_content`로 받되, 구글 시트면 `exportMimeType`을
   `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`로 지정한다.
   받은 base64를 풀어 `toms/data/운영시트.xlsx`로 저장한다.
   - 드라이브 연결이 없거나 파일을 못 찾으면 `toms/data/운영시트.xlsx`가 이미 있는지 본다. 없으면 멈추고 그 사실을 알린다. 데모 데이터로 대신하지 않는다.
2. **실행.** `cd toms && python -m toms_ops brief` (기준일은 한국 시간 오늘. 컨테이너 시계가 UTC면 `--today`로 지정한다.)
3. **올리기.** `toms/out/브리핑_<날짜>.html`을 드라이브의 `탐스 브리핑` 폴더에 올린다
   (`create_file`, `contentMimeType: text/html`, `disableConversionToGoogleType: true`). 폴더가 없으면 만든다.
4. **보고.** 아래 세 가지만 짧게 쓴다.
   - 긴급 건수와 내용 한 줄씩
   - 원장에게 전달할 건 (있을 때만)
   - 데이터 점검 건수

## 지킬 것

- 학생 이름·연락처는 드라이브와 이 세션 밖으로 내보내지 않는다. git에 커밋하지 않는다 (`toms/data`, `toms/out`은 무시 목록에 있다).
- 문자를 직접 보내지 않는다. 초안은 브리핑 안에 있고, 보내는 것은 관리자다.
- 규칙(`toms/config/rules.toml`)은 원장이 요청할 때만 고친다.
