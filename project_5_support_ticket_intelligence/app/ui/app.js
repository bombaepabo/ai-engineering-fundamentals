// app/ui/app.js

document.addEventListener("DOMContentLoaded", () => {
    // --- State Management ---
    const state = {
        activeTab: "tickets",
        tickets: [],
        activeTicket: null,
        articles: [],
        chatSessionId: "support-session-default",
        chatMessages: [],
        apiKey: localStorage.getItem("support_ai_api_key") || "dev-secret-key",
    };

    // --- DOM Elements ---
    const tabButtons = document.querySelectorAll(".nav-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    const apiStatusDot = document.querySelector("#api-status .status-dot");
    const apiStatusText = document.querySelector("#api-status .status-text");
    const apiStatusBadge = document.querySelector("#api-status");
    const settingsToggle = document.querySelector("#settings-toggle");
    
    // Modals
    const settingsModal = document.querySelector("#settings-modal");
    const closeSettingsModal = document.querySelector("#close-settings-modal");
    const saveSettingsBtn = document.querySelector("#save-settings-btn");
    const settingsApiKeyInput = document.querySelector("#settings-api-key");
    const ticketModal = document.querySelector("#ticket-modal");
    const closeTicketModal = document.querySelector("#close-ticket-modal");
    const cancelTicketModal = document.querySelector("#cancel-ticket-modal");
    const newTicketBtn = document.querySelector("#new-ticket-btn");
    const ticketForm = document.querySelector("#ticket-form");
    
    // Ticket Tab Elements
    const ticketListContainer = document.querySelector("#ticket-list-container");
    const workspaceEmptyState = document.querySelector("#workspace-empty-state");
    const ticketDetailsPane = document.querySelector("#ticket-details-pane");
    const detailTicketId = document.querySelector("#detail-ticket-id");
    const detailTicketSubject = document.querySelector("#detail-ticket-subject");
    const detailTicketStatus = document.querySelector("#detail-ticket-status");
    const detailTicketTime = document.querySelector("#detail-ticket-time");
    const detailTicketMessage = document.querySelector("#detail-ticket-message");
    const analyzeBtn = document.querySelector("#analyze-btn");
    
    // Ticket AI Elements
    const aiCategory = document.querySelector("#ai-category");
    const aiPriority = document.querySelector("#ai-priority");
    const aiSentiment = document.querySelector("#ai-sentiment");
    const aiConfidence = document.querySelector("#ai-confidence");
    const aiSummaryText = document.querySelector("#ai-summary-text");
    const aiMethod = document.querySelector("#ai-method");
    const copyReplyBtn = document.querySelector("#copy-reply-btn");
    const aiReplyText = document.querySelector("#ai-reply-text");
    const aiSourcesContainer = document.querySelector("#ai-sources-container");
    
    // Knowledge Tab Elements
    const knowledgeForm = document.querySelector("#knowledge-form");
    const articlesTableBody = document.querySelector("#articles-table-body");

    // Chat Tab Elements
    const chatSessionInput = document.querySelector("#chat-session-id");
    const chatMessagesContainer = document.querySelector("#chat-messages-container");
    const chatInputField = document.querySelector("#chat-input-field");
    const chatSendBtn = document.querySelector("#chat-send-btn");
    const chatTypingIndicator = document.querySelector("#chat-typing-indicator");

    // Toast
    const toastNotification = document.querySelector("#toast-notification");

    // Initialize Settings input field
    settingsApiKeyInput.value = state.apiKey;

    // --- Helper Functions ---
    
    // Show Toast Alert
    function showToast(message, isError = false) {
        toastNotification.textContent = message;
        toastNotification.style.borderColor = isError ? "var(--status-offline)" : "var(--accent-primary)";
        toastNotification.classList.remove("hidden");
        
        setTimeout(() => {
            toastNotification.classList.add("hidden");
        }, 3000);
    }

    // Fetch Helper with Auth Headers
    async function apiFetch(path, options = {}) {
        const headers = {
            "Content-Type": "application/json",
            "X-API-Key": state.apiKey,
            ...(options.headers || {}),
        };
        
        const response = await fetch(path, { ...options, headers });
        
        if (response.status === 401) {
            showToast("Unauthorized! Please check your API Key in Settings.", true);
            throw new Error("Unauthorized");
        }
        
        return response;
    }

    // Check API and Database Health
    async function checkApiHealth() {
        try {
            const res = await fetch("/health");
            if (!res.ok) throw new Error("Offline");
            
            const data = await res.json();
            if (data.status === "online" && data.database === "healthy") {
                apiStatusBadge.className = "status-badge online";
                apiStatusText.textContent = "API & DB Live";
            } else {
                apiStatusBadge.className = "status-badge degraded";
                apiStatusText.textContent = "Database Degraded";
            }
        } catch (e) {
            apiStatusBadge.className = "status-badge";
            apiStatusText.textContent = "Connection Failed";
        }
    }

    // --- Tab Switching ---
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            document.querySelector(`#tab-${targetTab}`).classList.add("active");
            
            state.activeTab = targetTab;
            
            // Refresh data based on active tab
            if (targetTab === "tickets") {
                loadTickets();
            } else if (targetTab === "knowledge") {
                loadArticles();
            }
        });
    });

    // --- Settings Drawer ---
    settingsToggle.addEventListener("click", () => settingsModal.classList.remove("hidden"));
    closeSettingsModal.addEventListener("click", () => settingsModal.classList.add("hidden"));
    saveSettingsBtn.addEventListener("click", () => {
        const newKey = settingsApiKeyInput.value.trim();
        if (newKey) {
            state.apiKey = newKey;
            localStorage.setItem("support_ai_api_key", newKey);
            showToast("API configurations saved successfully!");
            settingsModal.classList.add("hidden");
            
            // Refresh content with new key
            if (state.activeTab === "tickets") loadTickets();
            else if (state.activeTab === "knowledge") loadArticles();
        }
    });

    // --- Ticket Creation Dialog ---
    newTicketBtn.addEventListener("click", () => ticketModal.classList.remove("hidden"));
    closeTicketModal.addEventListener("click", () => ticketModal.classList.add("hidden"));
    cancelTicketModal.addEventListener("click", () => ticketModal.classList.add("hidden"));
    
    ticketForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const subject = document.querySelector("#ticket-subject").value.trim();
        const message = document.querySelector("#ticket-message").value.trim();
        
        try {
            const res = await apiFetch("/tickets", {
                method: "POST",
                body: JSON.stringify({ subject, message })
            });
            
            if (res.status === 201) {
                showToast("Ticket submitted successfully!");
                ticketForm.reset();
                ticketModal.classList.add("hidden");
                loadTickets();
            }
        } catch (e) {
            showToast("Failed to create ticket.", true);
        }
    });

    // --- Load Ticket Queue ---
    async function loadTickets() {
        try {
            const res = await apiFetch("/tickets");
            const data = await res.json();
            
            state.tickets = data;
            renderTicketsList();
        } catch (e) {
            ticketListContainer.innerHTML = `<li class="ticket-item-empty">Error loading queue. Check Auth key.</li>`;
        }
    }

    // Render Ticket Queue
    function renderTicketsList() {
        if (state.tickets.length === 0) {
            ticketListContainer.innerHTML = `<li class="ticket-item-empty">No tickets in the queue.</li>`;
            return;
        }

        ticketListContainer.innerHTML = "";
        
        // Sort tickets: newest first
        const sortedTickets = [...state.tickets].sort((a, b) => b.id - a.id);
        
        sortedTickets.forEach(ticket => {
            const li = document.createElement("li");
            li.className = `ticket-item ${state.activeTicket && state.activeTicket.id === ticket.id ? "active" : ""}`;
            
            const time = new Date(ticket.created_at).toLocaleDateString(undefined, {
                month: "short", day: "numeric"
            });

            li.innerHTML = `
                <div class="ticket-item-top">
                    <span class="ticket-item-id">#${ticket.id}</span>
                    <span class="ticket-item-status ${ticket.status}">${ticket.status}</span>
                </div>
                <h3>${escapeHtml(ticket.subject)}</h3>
                <div class="ticket-item-badges" id="badges-row-${ticket.id}">
                    <!-- Loaded dynamically if analyzed -->
                </div>
            `;
            
            li.addEventListener("click", () => selectTicket(ticket));
            ticketListContainer.appendChild(li);
            
            // If ticket is already analyzed, fetch metadata to display badges
            if (ticket.status === "analyzed") {
                loadMiniAnalysisBadges(ticket.id);
            }
        });
    }

    // Escape HTML to prevent injection
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Load Mini Analysis data for badges in list
    async function loadMiniAnalysisBadges(ticketId) {
        try {
            // Find analysis in db
            const res = await apiFetch(`/tickets/${ticketId}`);
            const ticketData = await res.json();
            
            if (ticketData.analyses && ticketData.analyses.length > 0) {
                const analysis = ticketData.analyses[0];
                const badgeRow = document.querySelector(`#badges-row-${ticketId}`);
                if (badgeRow) {
                    badgeRow.innerHTML = `
                        <span class="badge priority-${analysis.priority.toLowerCase()}">${analysis.priority}</span>
                        <span class="badge sentiment-${analysis.sentiment.toLowerCase()}">${analysis.sentiment}</span>
                        <span class="badge category-tag">${analysis.category}</span>
                    `;
                }
            }
        } catch (e) {
            console.error("Failed to load badges for ticket #", ticketId, e);
        }
    }

    // --- Select Ticket from Queue ---
    async function selectTicket(ticket) {
        state.activeTicket = ticket;
        
        // Update selection class in sidebar
        const listItems = document.querySelectorAll(".ticket-item");
        listItems.forEach(item => item.classList.remove("active"));
        
        // Refresh ticket details to make sure we have active references
        workspaceEmptyState.classList.add("hidden");
        ticketDetailsPane.classList.remove("hidden");
        
        detailTicketId.textContent = `#${ticket.id}`;
        detailTicketSubject.textContent = ticket.subject;
        detailTicketStatus.textContent = ticket.status;
        detailTicketStatus.className = `badge ${ticket.status}`;
        
        const timestamp = new Date(ticket.created_at).toLocaleString();
        detailTicketTime.textContent = timestamp;
        
        detailTicketMessage.textContent = ticket.message;
        
        // Reset analysis views
        aiCategory.textContent = "Pending";
        aiCategory.className = "badge";
        aiPriority.textContent = "Pending";
        aiPriority.className = "badge";
        aiSentiment.textContent = "Pending";
        aiSentiment.className = "badge";
        aiConfidence.textContent = "0.0";
        aiConfidence.className = "badge";
        aiSummaryText.innerHTML = `<span class="muted-placeholder">Run Analysis to generate AI summary...</span>`;
        aiMethod.textContent = "mock";
        aiReplyText.innerHTML = `<span class="muted-placeholder">Suggested email response will appear here after analysis.</span>`;
        copyReplyBtn.disabled = true;
        aiSourcesContainer.innerHTML = `<span class="muted-placeholder">No articles referenced yet.</span>`;
        
        // Re-render active item highlight
        renderTicketsList();
        
        // Load existing analysis if available
        loadTicketAnalysis(ticket.id);
    }

    // Load complete analysis details
    async function loadTicketAnalysis(ticketId) {
        try {
            const res = await apiFetch(`/tickets/${ticketId}`);
            const ticketData = await res.json();
            
            if (ticketData.analyses && ticketData.analyses.length > 0) {
                renderAnalysisData(ticketData.analyses[0]);
            }
        } catch (e) {
            console.error("Error loading analysis for ticket #", ticketId);
        }
    }

    // Render Analysis Report
    function renderAnalysisData(analysis) {
        aiCategory.textContent = analysis.category;
        aiCategory.className = "badge category-tag";
        
        aiPriority.textContent = analysis.priority;
        aiPriority.className = `badge priority-${analysis.priority.toLowerCase()}`;
        
        aiSentiment.textContent = analysis.sentiment;
        aiSentiment.className = `badge sentiment-${analysis.sentiment.toLowerCase()}`;
        
        aiConfidence.textContent = `${Math.round(analysis.confidence * 100)}%`;
        aiConfidence.className = "badge category-tag";
        
        aiSummaryText.textContent = analysis.summary;
        
        aiMethod.textContent = analysis.analysis_method;
        aiReplyText.textContent = analysis.suggested_reply;
        copyReplyBtn.disabled = false;
        
        // Render RAG sources
        if (analysis.sources && analysis.sources.length > 0) {
            aiSourcesContainer.innerHTML = "";
            analysis.sources.forEach(src => {
                const item = document.createElement("div");
                item.className = "source-item";
                item.innerHTML = `
                    <div class="source-title"><i class="fa-solid fa-file-invoice"></i> ${escapeHtml(src)}</div>
                `;
                aiSourcesContainer.appendChild(item);
            });
        } else {
            aiSourcesContainer.innerHTML = `<span class="muted-placeholder">No articles referenced.</span>`;
        }
    }

    // --- Run ML & LLM Analysis ---
    analyzeBtn.addEventListener("click", async () => {
        if (!state.activeTicket) return;
        
        const originalText = analyzeBtn.innerHTML;
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...`;
        
        try {
            const res = await apiFetch(`/tickets/${state.activeTicket.id}/analyze`, {
                method: "POST"
            });
            
            if (res.status === 201) {
                const analysisData = await res.json();
                showToast("Analysis complete!");
                renderAnalysisData(analysisData);
                
                // Update active status in state
                state.activeTicket.status = "analyzed";
                detailTicketStatus.textContent = "analyzed";
                detailTicketStatus.className = "badge analyzed";
                
                loadTickets(); // Reload sidebar to display badges
            }
        } catch (e) {
            showToast("Analysis failed. Check your LLM API Key.", true);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = originalText;
        }
    });

    // Copy Reply to Clipboard
    copyReplyBtn.addEventListener("click", () => {
        const text = aiReplyText.textContent;
        navigator.clipboard.writeText(text).then(() => {
            showToast("Copied draft response to clipboard!");
        }).catch(() => {
            showToast("Failed to copy text.", true);
        });
    });

    // --- TAB 2: KNOWLEDGE BASE HANDLERS ---
    
    // Load Policy Articles
    async function loadArticles() {
        try {
            const res = await apiFetch("/knowledge");
            const data = await res.json();
            state.articles = data;
            renderArticlesTable();
        } catch (e) {
            articlesTableBody.innerHTML = `<tr><td colspan="5" class="table-empty">Error loading articles.</td></tr>`;
        }
    }

    // Render Articles list
    function renderArticlesTable() {
        if (state.articles.length === 0) {
            articlesTableBody.innerHTML = `<tr><td colspan="5" class="table-empty">No policy documents published yet.</td></tr>`;
            return;
        }

        articlesTableBody.innerHTML = "";
        
        // Sort: newest first
        const sorted = [...state.articles].sort((a, b) => b.id - a.id);
        
        sorted.forEach(art => {
            const tr = document.createElement("tr");
            const date = new Date(art.created_at).toLocaleDateString();
            
            tr.innerHTML = `
                <td>#${art.id}</td>
                <td style="font-weight: 500;">${escapeHtml(art.title)}</td>
                <td><span class="analysis-method-badge">${art.content_type}</span></td>
                <td><span class="badge status-${art.status.toLowerCase()}">${art.status}</span></td>
                <td class="timestamp">${date}</td>
            `;
            articlesTableBody.appendChild(tr);
        });
    }

    // Publish Article Form submit
    knowledgeForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const title = document.querySelector("#article-title").value.trim();
        const content = document.querySelector("#article-content").value.trim();
        const sourceUrlInput = document.querySelector("#article-url").value.trim();
        
        const payload = {
            title,
            content,
            content_type: "text"
        };
        
        if (sourceUrlInput) {
            payload.source_url = sourceUrlInput;
        }

        const submitBtn = knowledgeForm.querySelector("button[type='submit']");
        const originalText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Vectorizing...`;
        
        try {
            const res = await apiFetch("/knowledge", {
                method: "POST",
                body: JSON.stringify(payload)
            });
            
            if (res.status === 201) {
                showToast("Policy published! Running background vectorizer...");
                knowledgeForm.reset();
                loadArticles();
                
                // Poll GCS indexing status for 5 seconds to verify index completion
                let attempts = 0;
                const pollInterval = setInterval(async () => {
                    attempts++;
                    await loadArticles();
                    
                    const isAllDone = state.articles.every(a => a.status === "indexed" || a.status === "failed");
                    if (isAllDone || attempts > 5) {
                        clearInterval(pollInterval);
                    }
                }, 1500);
            }
        } catch (e) {
            showToast("Failed to upload policy.", true);
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    });

    // --- TAB 3: STATEFUL CHAT CONVERSATION ---
    
    chatSessionInput.addEventListener("change", () => {
        const val = chatSessionInput.value.trim();
        if (val) {
            state.chatSessionId = val;
            // Clear message UI and reload session history
            chatMessagesContainer.innerHTML = "";
            loadChatHistory();
        }
    });

    // Fetch previous chats for active session ID
    async function loadChatHistory() {
        chatMessagesContainer.innerHTML = `<div class="msg-assistant" style="padding: 1rem; color: var(--text-muted); font-size: 0.8rem; text-align: center;">Loading session history...</div>`;
        
        try {
            // Note: Since there isn't a direct list endpoint, we get logs by submitting an empty chat interaction
            // or starting a conversation. Actually, the chat endpoint `/chat` retrieves history inside the backend.
            // Let's print the default helper first.
            chatMessagesContainer.innerHTML = `
                <div class="msg msg-assistant">
                    <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
                    <div class="msg-bubble">
                        Conversation context loaded for session <strong>"${state.chatSessionId}"</strong>. Feel free to ask a question!
                    </div>
                </div>
            `;
        } catch (e) {
            console.error("Error loading chat history");
        }
    }

    // Send chat message
    async function sendChatMessage() {
        const msgText = chatInputField.value.trim();
        if (!msgText) return;

        // Render User bubble immediately
        renderChatBubble(msgText, "user");
        chatInputField.value = "";
        
        // Show Typing indicator
        chatTypingIndicator.classList.remove("hidden");
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
        
        chatSendBtn.disabled = true;

        try {
            const res = await apiFetch("/chat", {
                method: "POST",
                body: JSON.stringify({
                    session_id: state.chatSessionId,
                    message: msgText
                })
            });
            
            if (res.status === 201) {
                const assistantReply = await res.json();
                renderChatBubble(assistantReply.content, "assistant", assistantReply.sources);
            }
        } catch (e) {
            renderChatBubble("Failed to reach the AI chat server. Make sure your API Keys are set correctly.", "assistant");
        } finally {
            chatTypingIndicator.classList.add("hidden");
            chatSendBtn.disabled = false;
            chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
        }
    }

    // Render Chat Bubble
    function renderChatBubble(content, role, sources = []) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `msg msg-${role}`;
        
        const avatarIcon = role === "user" ? "fa-user" : "fa-robot";
        
        let sourcesHtml = "";
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="msg-sources" style="font-size: 0.7rem; border-top: 1px dashed rgba(255, 255, 255, 0.05); margin-top: 0.5rem; padding-top: 0.25rem; color: var(--accent-secondary);">
                    Citations: ${sources.join(", ")}
                </div>
            `;
        }

        msgDiv.innerHTML = `
            <div class="msg-avatar"><i class="fa-solid ${avatarIcon}"></i></div>
            <div class="msg-bubble">
                ${escapeHtml(content).replace(/\n/g, "<br>")}
                ${sourcesHtml}
            </div>
        `;
        
        chatMessagesContainer.appendChild(msgDiv);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    // Send Event handlers
    chatSendBtn.addEventListener("click", sendChatMessage);
    chatInputField.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChatMessage();
    });

    // --- Initialize ---
    checkApiHealth();
    loadTickets();
    
    // Interval check health
    setInterval(checkApiHealth, 15000);
});
