import { useState, useEffect, useCallback } from 'react';
import ChatSidebar from '@/components/ChatSidebar';
import ChatInterface from '@/components/ChatInterface';
import { useRAG } from '@/hooks/useRAG';
import { useAuth } from '@/contexts/AuthContext';
import { chatsApi } from '@/lib/api';
import { toast } from 'sonner';

const Chat = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false); // Default: collapsed
  const [currentChatId, setCurrentChatId] = useState<number | null>(null);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [chatCreatedTrigger, setChatCreatedTrigger] = useState(0);
  const { token } = useAuth();
  const {
    documents,
    messages,
    isUploading,
    isLoading,
    uploadDocuments,
    sendQuery,
    startNewConversation,
    loadMessages,
    setOnChatCreated,
    threadId,
  } = useRAG(currentThreadId);

  // Trigger sidebar refresh when chat is created
  const handleChatCreated = useCallback(() => {
    setChatCreatedTrigger(prev => prev + 1);
  }, []);

  // Set callback for chat creation
  useEffect(() => {
    if (setOnChatCreated && typeof setOnChatCreated === 'function') {
      setOnChatCreated((chatId: number, threadId: string) => {
        setCurrentChatId(chatId);
        setCurrentThreadId(threadId);
        handleChatCreated();
      });
    }
  }, [setOnChatCreated, handleChatCreated]);

  // Handle new chat
  const handleNewChat = useCallback(() => {
    setCurrentChatId(null);
    setCurrentThreadId(null);
    startNewConversation();
  }, [startNewConversation]);

  // Handle chat selection - load messages from that chat
  const handleChatSelect = useCallback(async (chatId: number, threadId: string) => {
    // Update current chat/thread first
    setCurrentChatId(chatId);
    setCurrentThreadId(threadId);
    // Load messages for this chat (this will clear and replace existing messages)
    if (loadMessages) {
      await loadMessages(chatId, threadId);
    }
  }, [loadMessages]);

  // Handle file upload
  const handleUpload = useCallback(
    async (files: File[]) => {
      try {
        await uploadDocuments(files);
      } catch (error) {
        console.error('Failed to upload documents:', error);
      }
    },
    [uploadDocuments]
  );

  // Handle send query with thread_id
  const handleSendQuery = useCallback(
    async (message: string, images?: string[]) => {
      // Use current thread_id or generate new one
      const activeThreadId = currentThreadId || threadId;
      // Note: sendQuery signature may need to be updated if it has threadId parameter
      // For now, we'll pass images as the second parameter
      await sendQuery(message, images);
      // Always refresh sidebar after sending a message (for both new and existing chats)
      setChatCreatedTrigger(prev => prev + 1);
    },
    [currentThreadId, threadId, sendQuery]
  );

  return (
    <div className="flex h-screen overflow-hidden bg-[#0e0e0e]">
      <ChatSidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        currentChatId={currentChatId}
        onChatSelect={handleChatSelect}
        onNewChat={handleNewChat}
        onChatCreated={chatCreatedTrigger}
      />
      <ChatInterface
        messages={messages}
        onSend={handleSendQuery}
        isLoading={isLoading}
        hasDocuments={documents.length > 0}
        onMenuClick={() => setSidebarOpen(prev => !prev)}
        onUpload={handleUpload}
        isUploading={isUploading}
        sidebarOpen={sidebarOpen}
      />
    </div>
  );
};

export default Chat;

