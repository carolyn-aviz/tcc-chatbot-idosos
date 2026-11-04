// Seleção dos elementos do DOM
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');

// Gera um ID de sessão único para esta aba/navegador
const sessionId = 'sessao_' + Math.random().toString(36).substring(2, 11);

// Função para inserir mensagens na interface com suporte a quebras de linha e negrito
function adicionarMensagem(texto, remetente) {
  const divMsg = document.createElement('div');
  divMsg.classList.add('msg', remetente);
  
  // 🌟 Converte as quebras de linha (\n) do Python em <br> do HTML
  // E substitui **negrito** por <strong>negrito</strong> se necessário
  let textoFormatado = texto
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  divMsg.innerHTML = textoFormatado;
  
  chatMessages.appendChild(divMsg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Função de integração com a API do Back-End (FastAPI)
async function processarEnvio() {
  const texto = userInput.value.trim();
  if (!texto) return;

  // 1. Exibe a mensagem do usuário na tela
  adicionarMensagem(texto, 'user');
  userInput.value = '';
  userInput.focus();

  try {
    // 2. Requisição POST para o FastAPI
    const resposta = await fetch('http://127.0.0.1:8000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        mensagem: texto,
        session_id: sessionId 
      })
    });
    
    if (!resposta.ok) {
      throw new Error('Erro na resposta do servidor');
    }

    const dados = await resposta.json();
    adicionarMensagem(dados.resposta, 'bot');

  } catch (erro) {
    console.error('Erro de conexão:', erro);
    adicionarMensagem('Desculpe, ocorreu um erro ao conectar ao servidor. Verifique se o FastAPI está a rodar.', 'bot');
  }
}

// Evento de clique no botão "Enviar"
sendBtn.addEventListener('click', processarEnvio);

// Evento para enviar também ao pressionar a tecla "Enter" no teclado
userInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    processarEnvio();
  }
});