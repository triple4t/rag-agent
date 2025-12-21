import { useState } from 'react';
import { ChevronDown, ChevronUp, FileText } from 'lucide-react';
import Avatar from './Avatar';
import type { Message } from '@/types';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage = ({ message }: ChatMessageProps) => {
  const [showSources, setShowSources] = useState(false);

  return (
    <div
      className={`w-full py-6 animate-fade-in ${
        message.role === 'assistant' ? 'bg-message-assistant' : 'bg-message-user'
      }`}
    >
      <div className="max-w-3xl mx-auto px-4 flex gap-4">
        <Avatar type={message.role} />
        <div className="flex-1 min-w-0">
          <div className="text-foreground whitespace-pre-wrap break-words leading-relaxed">
            {message.content}
          </div>

          {message.sources && message.sources.length > 0 && (
            <div className="mt-4">
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <FileText className="w-4 h-4" />
                <span>{message.sources.length} sources</span>
                {showSources ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>

              {showSources && (
                <div className="mt-3 space-y-2">
                  {message.sources.map((source, index) => (
                    <div
                      key={index}
                      className="p-3 rounded-lg bg-accent border border-border text-sm"
                    >
                      <div className="font-medium text-foreground mb-1 flex items-center gap-2">
                        <span>{source.filename || `Document ${source.doc_id.slice(0, 8)}`}</span>
                        {source.score !== undefined && (
                          <span className="text-xs text-muted-foreground">
                            (Score: {source.score.toFixed(3)})
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-muted-foreground mb-1">
                        Chunk {source.chunk_idx + 1}
                      </div>
                      <div className="text-muted-foreground line-clamp-3">
                        {source.content}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {message.role === 'assistant' && message.qualityScore !== undefined && (
            <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
              <span>Quality: {Math.round(message.qualityScore * 100)}%</span>
              {message.latency && <span>{message.latency.toFixed(2)}s</span>}
              {message.cost && <span>${message.cost.toFixed(4)}</span>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
