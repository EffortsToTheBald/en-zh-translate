// src/components/Translator.js
import React, { useState } from 'react';

const Translator = () => {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');


  // const getBackendUrl = () => {
  //   if (typeof window !== 'undefined' && window.APP_CONFIG?.backend) {
  //     const { serviceName, namespace, port } = window.APP_CONFIG.backend;
  //     // Kubernetes Service DNS
  //     return `http://${serviceName}.${namespace}.svc.cluster.local:${port}`;
  //   }
  //   // 开发 fallback（可选）
  //   return 'http://192.168.1.19:8000';
  // };
  //  const API_URL = `${getBackendUrl()}/translate`;
 

  // 后端 API 地址（开发时）
  var API_URL = "/api/translate";

  const handleTranslate = async () => {
    // 清空上一次结果和错误
    setError('');
    setOutputText('');

    // 检查输入
    if (!inputText.trim()) {
      setError('请输入英文句子');
      return;
    }

    setLoading(true);

    try {
      console.log("API_URL",API_URL)
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText.trim(),
          temperature: 0.8, // 可调整，范围 0.1～2.0
        }),
      });

      if (!response.ok) {
        // 尝试读取错误详情
        let errorMsg = `请求失败 (${response.status})`;
        try {
          const errorData = await response.json();
          errorMsg = errorData.detail || errorMsg;
        } catch (e) {
          // 忽略 JSON 解析失败
        }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      setOutputText(data.translation || '');
    } catch (err) {
      console.error('Translation error:', err);
      setError(err.message || '翻译服务出错，请稍后再试');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setInputText('');
    setOutputText('');
    setError('');
  };

  return (
    <div style={{ padding: '2rem', maxWidth: '700px', margin: '0 auto', fontFamily: 'Arial, sans-serif' }}>
      <h1>🔤 英文 → 中文 翻译器</h1>
      <p>基于 Transformer 模型（PyTorch）</p>

      <div style={{ marginBottom: '1rem' }}>
        <label htmlFor="english-input">英文输入:</label>
        <br />
        <textarea
          id="english-input"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="例如: A dog is running in the park"
          rows="4"
          cols="70"
          style={{
            width: '100%',
            padding: '8px',
            marginTop: '4px',
            borderRadius: '4px',
            border: '1px solid #ccc',
          }}
        />
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <button
          onClick={handleTranslate}
          disabled={loading || !inputText.trim()}
          style={{
            backgroundColor: loading ? '#ccc' : '#007bff',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            marginRight: '8px',
          }}
        >
          {loading ? '翻译中...' : '翻译'}
        </button>
        <button
          onClick={handleClear}
          style={{
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          清空
        </button>
      </div>

      {error && (
        <div
          style={{
            color: 'red',
            marginBottom: '1rem',
            padding: '8px',
            backgroundColor: '#ffebee',
            borderRadius: '4px',
          }}
        >
          ❌ {error}
        </div>
      )}

      {outputText && (
        <div>
          <label>中文翻译:</label>
          <div
            style={{
              marginTop: '4px',
              padding: '12px',
              backgroundColor: '#f8f9fa',
              border: '1px solid #ddd',
              borderRadius: '4px',
              minHeight: '60px',
              fontSize: '1.1em',
            }}
          >
            {outputText}
          </div>
        </div>
      )}
    </div>
  );
};

export default Translator;