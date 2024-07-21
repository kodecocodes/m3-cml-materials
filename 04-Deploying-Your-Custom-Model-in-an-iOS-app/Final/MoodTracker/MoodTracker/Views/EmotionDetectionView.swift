/// Copyright (c) 2024 Kodeco Inc.
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
/// distribute, sublicense, create a derivative work, and/or sell copies of the
/// Software in any work that is designed, intended, or marketed for pedagogical or
/// instructional purposes related to programming, coding, application development,
/// or information technology.  Permission for such use, copying, modification,
/// merger, publication, distribution, sublicensing, creation of derivative works,
/// or sale is expressly withheld.
///
/// This project and source code may use libraries or frameworks that are
/// released under various Open-Source licenses. Use of those libraries and
/// frameworks are governed by their own individual licenses.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

import SwiftUI

struct EmotionDetectionView: View {
  @State private var image: UIImage?
  @State private var emotion: String?
  @State private var accuracy: String?
  @State private var isShowingImagePicker = false
  @State private var sourceType: UIImagePickerController.SourceType = .photoLibrary
  @State private var showSourceTypeActionSheet = false
  let classifier = EmotionClassifier()

  var body: some View {
    VStack(spacing: 20) {
      ImageDisplayView(image: $image, showSourceTypeActionSheet: $showSourceTypeActionSheet)

      if let emotion = emotion, let accuracy = accuracy {
        EmotionResultView(emotion: emotion, accuracy: accuracy)
      }

      ActionButtonsView(image: $image, classifyImage: classifyImage, reset: reset)
    }
    .navigationTitle("Emotion Detection")
    .actionSheet(isPresented: $showSourceTypeActionSheet) {
      ActionSheet(title: Text("Select Image Source"), message: nil, buttons: [
        .default(Text("Camera")) {
          self.sourceType = .camera
          self.isShowingImagePicker = true
        },
        .default(Text("Photo Library")) {
          self.sourceType = .photoLibrary
          self.isShowingImagePicker = true
        },
        .cancel()
      ])
    }
    .sheet(isPresented: $isShowingImagePicker) {
      ImagePicker(image: self.$image, sourceType: self.$sourceType)
    }
  }

  func classifyImage() {
    if let image = self.image {
      classifier.classify(image: image) { emotion, confidence in
        self.emotion = emotion ?? "Unknown"
        self.accuracy = String(format: "%.2f%%", (confidence ?? 0) * 100.0)
      }
    }
  }

  func reset() {
    self.image = nil
    self.emotion = nil
    self.accuracy = nil
  }
}


#Preview {
  EmotionDetectionView()
}
