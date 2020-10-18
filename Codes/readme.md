codes!

- readme.md: 본 화일입니다.

- SleepDataPreparation_and_Classification.ipynb : 데이터 다운로드 부터 모든 처리가 통합된 버전
   - download_physionet.sh 사용

- simpleFC_with_Relu.ipynb : Coogle Drive + Fully Connected + Relu 모델
   - best epoch : 22/2000 / accuracy : 56.344460%
   
- simpleFC_with_Relu2.ipynb : Dropout 추가. 그러나, 성능향상은 없어보임

- SleepNet_with_Convolution.ipynb : Convolution Layer 추가

>>        self.conv1d_1 = nn.Conv1d(1,16, kernel_size=200, stride=50)
>>        self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=20, stride=5)
>>        self.fc1 = nn.Linear(256,160)
>>        self.fc2 = nn.Linear(160,out_channel) 
>>
>>        self.ReLU = nn.ReLU()
>>        self.dropout = nn.Dropout(p=0.3)
>>
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
>>


