# 10개의 판별 모델을 만들어서 competition 벌이는 대환장 소스코드!

** 전역변수 및 파라미터는 var_parameters.py에서 관리 **
** 각 폴더마다 반복문에 대해서 안전장치(시험삼아 해보기) 있으니까 잘 사용할 것 ** 

1. 폴더 구성
C:\data\LPD_competition 아래에 폴더 만들기:
- modelcheckpoint
- npy
- h5

2. 순서대로 파일 run

- 1_augment_img.py: 이미지 개수 증폭시키는 기능
- 1_deaugment_img.py: 이미지 증폭 이전으로 되돌려주는 기능
- 2_mk_npy.py: npy 파일 만들어줌


오리지널 파일 제외하고 다 없애주는 파일 만들어야겠다