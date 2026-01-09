import { useRef, useEffect } from 'react';
import { Menu } from 'lucide-react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import WelcomeScreen from './WelcomeScreen';
import TypingIndicator from './TypingIndicator';
import Avatar from './Avatar';
import type { Message } from '@/types';

interface ChatInterfaceProps {
  messages: Message[];
  onSend: (message: string, images?: string[]) => void;
  isLoading: boolean;
  hasDocuments: boolean;
  onMenuClick: () => void;
}

const ChatInterface = ({
  messages,
  onSend,
  isLoading,
  hasDocuments,
  onMenuClick,
}: ChatInterfaceProps) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Mobile header */}
      <div className="lg:hidden flex items-center gap-3 p-4 border-b border-border">
        <button
          onClick={onMenuClick}
          className="p-2 text-muted-foreground hover:text-foreground transition-colors"
        >
          <Menu className="w-5 h-5" />
        </button>
        <span className="font-medium text-foreground">RAG Assistant</span>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {messages.length === 0 ? (
          <WelcomeScreen
            hasDocuments={hasDocuments}
            onSuggestionClick={onSend}
          />
        ) : (
          <div className="pb-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="w-full py-6 bg-message-assistant">
                <div className="max-w-3xl mx-auto px-4 flex gap-4">
                  <Avatar type="assistant" />
                  <TypingIndicator />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <ChatInput onSend={onSend} disabled={isLoading} />
    </div>
  );
};

export default ChatInterface;
