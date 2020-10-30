codes!

- readme.md: 본 화일입니다.

- SleepDataPreparation_and_Classification.ipynb : 데이터 다운로드 부터 모든 처리가 통합된 버전
   - download_physionet.sh 사용

- simpleFC_with_Relu.ipynb : Coogle Drive + Fully Connected + Relu 모델
   - best epoch : 22/2000 / accuracy : 56.344460%
   
- simpleFC_with_Relu2.ipynb : Dropout 추가. 그러나, 성능향상은 없어보임

- SleepNet_with_Convolution.ipynb : Convolution Layer 추가

```python
>>    def forward(self, input):
>>        out = self.conv1d_1(input)
>>        out = self.ReLU(out)
>>        out = self.conv1d_2(out)
>>        out = self.ReLU(out)
>>        out = torch.flatten(out, 1)
>>        out = self.fc1(out)
>>        out = self.ReLU(out)
>>        out = self.fc2(out)
>>        out = self.ReLU(out)   
```


