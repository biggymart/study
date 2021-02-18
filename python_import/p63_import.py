from test_import import p62_import
# anaconda3 파일 아래 폴더 만들어서 임포트 가능한 것임

p62_import.sum2()

print("=============================")

from test_import.p62_import import sum2
sum2()