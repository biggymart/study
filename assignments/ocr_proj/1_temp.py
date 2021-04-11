def draw_image(myString):
    width=500
    height=100
    back_ground_color=(255,255,255)
    font_size=10
    font_color=(0,0,0)
    unicode_text = myString
    im  =  Image.new ( "RGB", (width,height), back_ground_color )
    draw  =  ImageDraw.Draw (im)
    unicode_font = ImageFont.truetype("arial.ttf", font_size)
    draw.text ( (10,10), unicode_text, font=unicode_font, fill=font_color )
    im.save("text.jpg")
    if cv2.waitKey(0)==ord('q'):
        cv2.destroyAllWindows()