import os
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.properties import ObjectProperty, StringProperty
from PIL import Image as PILImage
from ImageCaption.caption import preprocessImage, GenerateSpeech

Builder.load_string('''
<RootWidget>:
    orientation: 'vertical'
    padding: '10dp'
    spacing: '10dp'

    Image:
        id: img
        source: ''

    Label:
        id: caption_label
        text: ''
        size_hint_y: None
        height: '50dp'

    Button:
        text: 'Select Image'
        on_release: root.select_image()

    Button:
        text: 'Predict'
        on_release: root.predict_caption()
    Label:
        id: predict_label
        text: ''
        size_hint_y: None
        height: '50dp'
''')

class RootWidget(BoxLayout):
    img_path = StringProperty('')
    caption_text = StringProperty('')

    def select_image(self):
        from plyer import filechooser

        filters = ['*.jpg', '*.png']
        file_path = filechooser.open_file(filters=filters)[0]

        self.img_path = file_path

        self.ids.img.source = file_path

    def predict_caption(self):
        if not self.img_path:
            return

        image = PILImage.open(self.img_path)
        image = image.resize((224, 224))
        caption = preprocessImage(image)
        GenerateSpeech(caption)

        self.caption_text = caption
        self.ids.predict_label.text = caption

class ImageCaptionApp(App):
    def build(self):
        return RootWidget()

if __name__ == '__main__':
    from kivy.utils import platform

    if platform == 'android':
        from android import AndroidService

        service = AndroidService('Image Caption', 'running')
        service.start('service started')
        ImageCaptionApp().run()
        service.stop()
    else:
        ImageCaptionApp().run()
