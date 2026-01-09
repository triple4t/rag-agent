import { Plus, MessageSquare, Trash2, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';

interface Chat {
  id: number;
  title: string;
  thread_id: string;
  created_at: string;
  updated_at: string;
}

interface ChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  currentChatId: number | null;
  onChatSelect: (chatId: number, threadId: string) => void;
  onNewChat: () => void;
  onChatCreated?: () => void;  // Callback to refresh when new chat is created
}

const ChatSidebar = ({
  isOpen,
  onClose,
  currentChatId,
  onChatSelect,
  onNewChat,
  onChatCreated,
}: ChatSidebarProps) => {
  const { token } = useAuth();
  const [chats, setChats] = useState<Chat[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (token) {
      fetchChats();
    }
  }, [token]);

  const fetchChats = async () => {
    if (!token) return;
    
    setIsLoading(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/chats`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setChats(data.chats);
      }
    } catch (error) {
      console.error('Failed to fetch chats:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Refresh chats when onChatCreated changes (triggered from parent)
  useEffect(() => {
    if (onChatCreated !== undefined && onChatCreated !== null && token) {
      // Small delay to ensure backend has saved the chat
      const timer = setTimeout(() => {
        fetchChats();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [onChatCreated, token]);

  const deleteChat = async (chatId: number) => {
    if (!token) return;
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/chats/${chatId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (response.ok) {
        setChats(chats.filter(chat => chat.id !== chatId));
        if (currentChatId === chatId) {
          onNewChat();
        }
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
    }
  };

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 w-[260px] bg-[#171717] flex flex-col border-r border-[#2a2a2a] transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2.5 border-b border-[#2a2a2a]">
          <h2 className="text-sm font-medium text-[#e5e7eb]">Your chats</h2>
          <button
            onClick={onClose}
            className="p-1.5 text-[#9ca3af] hover:text-[#e5e7eb] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="px-3 py-2 border-b border-[#2a2a2a]">
          <button
            onClick={onNewChat}
            className="w-full flex items-center gap-2 px-3 py-2 bg-[#1a1a1a] hover:bg-[#2a2a2a] text-[#e5e7eb] rounded-lg transition-colors text-sm"
          >
            <Plus className="w-4 h-4" />
            New chat
          </button>
        </div>

        {/* Chats List */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-1.5 py-1">
          {isLoading ? (
            <div className="text-center text-[#9ca3af] py-8 text-sm">Loading chats...</div>
          ) : chats.length > 0 ? (
            <div className="space-y-0.5">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`group flex items-center gap-2 px-2.5 py-2 rounded-md cursor-pointer transition-colors ${
                    currentChatId === chat.id
                      ? 'bg-[#2a2a2a] text-[#e5e7eb]'
                      : 'text-[#9ca3af] hover:bg-[#1a1a1a]'
                  }`}
                  onClick={() => onChatSelect(chat.id, chat.thread_id)}
                >
                  <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
                  <span className="flex-1 truncate text-sm">{chat.title}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteChat(chat.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-[#2a2a2a] rounded transition-opacity"
                  >
                    <Trash2 className="w-3.5 h-3.5 text-[#9ca3af] hover:text-red-400" />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <MessageSquare className="w-10 h-10 text-[#2a2a2a] mb-3" />
              <p className="text-sm text-[#9ca3af]">No chats yet</p>
            </div>
          )}
        </div>
      </aside>
    </>
  );
};

export default ChatSidebar;

