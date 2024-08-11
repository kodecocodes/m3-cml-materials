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
  @StateObject private var viewModel = EmotionDetectionViewModel()
  @State private var isShowingImagePicker = false
  @State private var sourceType: UIImagePickerController.SourceType = .photoLibrary
  @State private var showSourceTypeActionSheet = false
  @State private var feedbackLabel: String = ""  // New state for user feedback

  var body: some View {
    VStack(spacing: 20) {
      ImageDisplayView(image: $viewModel.image, showSourceTypeActionSheet: $showSourceTypeActionSheet)

      if let emotion = viewModel.emotion, let accuracy = viewModel.accuracy {
        EmotionResultView(emotion: emotion, accuracy: accuracy)

        TextField("Enter Correct Label", text: $feedbackLabel)
          .textFieldStyle(RoundedBorderTextFieldStyle())
          .padding()

        Button("Update Model with Correct Label") {
          if let image = viewModel.image {
            viewModel.updateModel(with: image, label: feedbackLabel)
          }
        }
        .disabled(feedbackLabel.isEmpty || viewModel.isModelUpdating)
        .padding()

        if viewModel.isModelUpdating {
          ProgressView("Updating Model...")
        }
      }

      ActionButtonsView(image: $viewModel.image, classifyImage: viewModel.classifyImage, reset: viewModel.reset)
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
      ImagePicker(image: self.$viewModel.image, sourceType: self.$sourceType)
    }
  }
}


#Preview {
  EmotionDetectionView()
}

