'use client'

import { countErrors } from '@/lib/count-evaluation'
import { InferenceResult } from './types'

const format = (value: number) => new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 }).format(value)

export function CountComparison({ results, actual, visibleCount }: {
  results: InferenceResult; actual: number | null; visibleCount: number
}) {
  const predicted = results.predicted_count ?? (results.modelType === 'detection' ? visibleCount : null)
  const errors = actual !== null && predicted !== null ? countErrors(predicted, actual) : null
  const rawErrors = actual !== null && results.count_raw !== undefined ? countErrors(results.count_raw, actual) : null
  const isCalibrated = results.countCalibrated ?? (results.count_raw !== undefined)
  return <section className="count-comparison" aria-label="So sánh số quả dự đoán với thực tế" aria-live="polite">
    <div className="comparison-heading">
      <div><p className="studio-eyebrow">ĐÁNH GIÁ SỐ ĐẾM</p><h3>Dự đoán so với thực tế</h3></div>
      <span className="comparison-status">{actual === null ? 'Chưa có ground truth' : 'Đã nạp ground truth'}</span>
    </div>
    <div className="comparison-cards">
      <div className="comparison-card comparison-prediction">
        <p>Model dự đoán</p>
        <div className="comparison-value">{predicted === null ? '—' : format(predicted)} <span>quả</span></div>
        <small>{isCalibrated ? 'Số đếm sau hiệu chỉnh' : results.densityMap ? 'Ước lượng từ mật độ' : 'Số đếm đầu ra'}</small>
      </div>
      <div className="comparison-card comparison-actual">
        <p>Thực tế từ mask / label</p>
        <div className="comparison-value">{actual === null ? '—' : format(actual)} <span>quả</span></div>
        <small>{actual === null ? 'Chọn mask hoặc label ở bảng cấu hình' : 'Số đối tượng trong ground truth đã tải'}</small>
      </div>
      <div className={`comparison-card comparison-difference ${errors && errors.absolute === 0 ? 'is-equal' : ''}`}>
        <p>Chênh lệch số đếm</p>
        <div className="comparison-value">{errors ? format(errors.absolute) : '—'} <span>quả</span></div>
        <small>{errors ? errors.signed === 0 ? 'Số đếm bằng thực tế' : `${errors.signed > 0 ? 'Dự đoán nhiều hơn' : 'Dự đoán ít hơn'} ${format(errors.absolute)} quả` : 'Có ground truth sẽ tính sai số'}</small>
      </div>
    </div>
    {errors && <p className="comparison-verdict">
      Dự đoán <strong>{format(predicted!)}</strong> / thực tế <strong>{format(actual!)}</strong> quả.
      {' '}Sai số tương đối: <strong>{errors.percentage === null ? 'Không xác định (thực tế = 0)' : `${format(errors.percentage)}%`}</strong>.
      {errors.absolute === 0 && ' Số lượng bằng nhau không có nghĩa là mọi quả đều được nhận diện đúng vị trí.'}
    </p>}
    {isCalibrated && results.count_raw !== undefined && <div className="comparison-raw">
      <div><span>Đếm trực tiếp từ box</span><strong>{format(results.count_raw)} quả</strong></div>
      <div><span>Sau hiệu chỉnh</span><strong>{predicted === null ? '—' : format(predicted)} quả</strong></div>
      {rawErrors && <div><span>Chênh lệch box với thực tế</span><strong>{rawErrors.signed === 0 ? 'Bằng số lượng' : `${rawErrors.signed < 0 ? 'Ít hơn' : 'Nhiều hơn'} ${format(rawErrors.absolute)} quả`}</strong></div>}
      <p>Hiệu chỉnh chỉ thay đổi số đếm, không tạo thêm bounding box. Chênh lệch số lượng không phải số quả bỏ sót (FN); muốn xác định FN cần ghép box dự đoán với ground truth theo vị trí.</p>
    </div>}
  </section>
}
