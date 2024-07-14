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

  let classifier = EmotionClassifier()

  var body: some View {
    VStack {
      if let image = image {
        Image(uiImage: image)
          .resizable()
          .scaledToFit()
          .frame(width: 300, height: 300)
          .onTapGesture {
            self.isShowingImagePicker = true
          }
          .sheet(isPresented: $isShowingImagePicker) {
            ImagePicker(image: self.$image)
          }
      } else {
        Text("Tap to Upload an Image")
          .foregroundColor(.gray)
          .frame(width: 300, height: 300)
          .background(Color.black.opacity(0.1))
          .onTapGesture {
            self.isShowingImagePicker = true
          }
          .sheet(isPresented: $isShowingImagePicker) {
            ImagePicker(image: self.$image)
          }
      }

      if let emotion = emotion {
        Text("Detected Emotion: \(emotion)")
          .font(.title)
          .padding()
      }

      if let accuracy = accuracy {
        Text("Accuracy: \(accuracy)")
          .padding(.bottom)
      }

      Button(action: {
        self.classifyImage()
      }) {
        Text("Detect Emotion")
      }
      .padding()

      Button(action: {
        self.image = nil
        self.emotion = nil
        self.accuracy = nil
      }) {
        Text("Upload Another Image")
      }
      .padding()
      .disabled(image == nil)
    }
    .navigationTitle("Emotion Detection")
  }
  
  func classifyImage() {
    if let image = self.image {
      classifier.classify(image: image) { emotion, confidence in
        self.emotion = emotion ?? "Unknown"
        self.accuracy = String(format: "%.2f%%", (confidence ?? 0) * 100.0)
      }
    }
  }
}


#Preview {
  EmotionDetectionView()
}
