// static/chat.js
const socket = io();
const messagesDiv = document.getElementById('messages');
const inputBox   = document.getElementById('msgInput');
const sendBtn    = document.getElementById('sendBtn');

function append(html, role) {
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.innerHTML = html;
  messagesDiv.appendChild(el);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
function showTyping() {
  const el = document.createElement('div');
  el.id = 'typing';
  el.className = 'message bot typing';
  messagesDiv.appendChild(el);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
function hideTyping() { document.getElementById('typing')?.remove(); }

function sendMessage() {
  const txt = inputBox.value.trim();
  if (!txt) return;
  append(txt, 'user');
  inputBox.value = '';
  showTyping();
  socket.emit('user_message', {text: txt});
}

sendBtn.onclick = sendMessage;
inputBox.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });

socket.on('bot_reply', data => {
  hideTyping();
  append(data.html, 'bot');
});
