import { useState } from 'react';
import { ChevronDown, ChevronUp, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
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
          {/* Display images if present */}
          {message.images && message.images.length > 0 && (
            <div className="mb-3">
              <div className="flex flex-wrap gap-3">
                {message.images.map((image, index) => (
                  <div key={index} className="relative group">
                    <img
                      src={image}
                      alt={`Uploaded image ${index + 1}`}
                      className="max-w-xs max-h-64 rounded-lg border-2 border-border object-contain bg-accent/50 hover:border-primary/50 transition-colors cursor-pointer"
                      onClick={() => {
                        // Open image in new tab on click
                        const newWindow = window.open();
                        if (newWindow) {
                          newWindow.document.write(`<img src="${image}" style="max-width: 100%; height: auto;" />`);
                        }
                      }}
                    />
                    <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                      Click to view full size
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {message.images.length} image{message.images.length > 1 ? 's' : ''} attached
              </p>
            </div>
          )}
          
          <div className="text-foreground break-words leading-relaxed prose prose-invert prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // Style links
                a: ({ node, ...props }) => (
                  <a
                    {...props}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:text-blue-300 underline underline-offset-2 transition-colors"
                  />
                ),
                // Style headings
                h1: ({ node, ...props }) => (
                  <h1 {...props} className="text-xl font-bold mt-4 mb-2 text-foreground" />
                ),
                h2: ({ node, ...props }) => (
                  <h2 {...props} className="text-lg font-semibold mt-3 mb-2 text-foreground" />
                ),
                h3: ({ node, ...props }) => (
                  <h3 {...props} className="text-base font-semibold mt-2 mb-1 text-foreground" />
                ),
                // Style lists
                ul: ({ node, ...props }) => (
                  <ul {...props} className="list-disc list-inside my-2 space-y-1 ml-4" />
                ),
                ol: ({ node, ...props }) => (
                  <ol {...props} className="list-decimal list-inside my-2 space-y-1 ml-4" />
                ),
                li: ({ node, ...props }) => (
                  <li {...props} className="my-1" />
                ),
                // Style paragraphs
                p: ({ node, ...props }) => (
                  <p {...props} className="my-2" />
                ),
                // Style bold text
                strong: ({ node, ...props }) => (
                  <strong {...props} className="font-semibold text-foreground" />
                ),
                // Style code blocks
                code: ({ node, inline, ...props }: any) =>
                  inline ? (
                    <code
                      {...props}
                      className="bg-accent px-1.5 py-0.5 rounded text-sm font-mono"
                    />
                  ) : (
                    <code
                      {...props}
                      className="block bg-accent p-3 rounded text-sm font-mono overflow-x-auto my-2"
                    />
                  ),
              }}
            >
              {message.content}
            </ReactMarkdown>
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
                  {message.sources.map((source, index) => {
                    // Check if doc_id is a URL (for web search sources)
                    const isUrl = source.doc_id?.startsWith('http://') || source.doc_id?.startsWith('https://');
                    const sourceUrl = isUrl ? source.doc_id : null;
                    
                    return (
                      <div
                        key={index}
                        className="p-3 rounded-lg bg-accent border border-border text-sm"
                      >
                        <div className="font-medium text-foreground mb-1 flex items-center gap-2 flex-wrap">
                          {sourceUrl ? (
                            <a
                              href={sourceUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-400 hover:text-blue-300 underline underline-offset-2 transition-colors"
                            >
                              {source.filename || sourceUrl}
                            </a>
                          ) : (
                            <span>{source.filename || `Document ${source.doc_id?.slice(0, 8) || 'Unknown'}`}</span>
                          )}
                          {source.score !== undefined && (
                            <span className="text-xs text-muted-foreground">
                              (Score: {source.score.toFixed(4)})
                            </span>
                          )}
                        </div>
                        {!isUrl && (
                          <div className="text-xs text-muted-foreground mb-1">
                            {source.page_number ? (
                              <>Page {source.page_number} • Chunk {source.chunk_idx + 1}</>
                            ) : (
                              <>Chunk {source.chunk_idx + 1}</>
                            )}
                          </div>
                        )}
                        <div className="text-muted-foreground line-clamp-3">
                          {source.content}
                        </div>
                      </div>
                    );
                  })}
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
