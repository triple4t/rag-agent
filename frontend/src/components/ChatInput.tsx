import { useState, useRef, useEffect, KeyboardEvent, ChangeEvent } from 'react';
import { Send, X, Image as ImageIcon } from 'lucide-react';
import { toast } from 'sonner';

interface ChatInputProps {
  onSend: (message: string, images?: string[]) => void;
  disabled?: boolean;
}

const ChatInput = ({ onSend, disabled }: ChatInputProps) => {
  const [message, setMessage] = useState('');
  const [images, setImages] = useState<string[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleImageSelect = (e: ChangeEvent<HTMLInputElement>) => {
    console.log('handleImageSelect called', e);
    console.log('Event target:', e.target);
    console.log('Event target files:', e.target?.files);
    
    const files = e.target?.files;
    if (!files || files.length === 0) {
      console.log('No files selected - files object:', files);
      toast.warning('No files were selected. Please try again.');
      return;
    }

    console.log(`Selected ${files.length} file(s)`);
    console.log('File list:', Array.from(files).map(f => ({ name: f.name, type: f.type, size: f.size })));
    const imagePromises: Promise<string>[] = [];
    let validImageCount = 0;
    let skippedCount = 0;
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      console.log(`Processing file ${i + 1}: ${file.name}, type: ${file.type || 'unknown'}, size: ${(file.size / 1024).toFixed(2)}KB`);
      
      // Check file type - handle both MIME type and file extension
      const fileExtension = file.name.toLowerCase().split('.').pop() || '';
      const imageExtensions = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg', 'ico'];
      const isImageByExtension = imageExtensions.includes(fileExtension);
      const isImageByMime = file.type && file.type.startsWith('image/');
      
      if (!isImageByMime && !isImageByExtension) {
        console.warn(`Skipping non-image file: ${file.name} (type: ${file.type || 'unknown'}, extension: ${fileExtension})`);
        toast.error(`${file.name} is not a valid image file. Please select PNG, JPEG, GIF, WebP, BMP, or SVG files.`);
        skippedCount++;
        continue;
      }

      // Check file size (limit to 10MB per image)
      const maxSize = 10 * 1024 * 1024; // 10MB
      if (file.size > maxSize) {
        console.warn(`Image ${file.name} is too large (${(file.size / 1024 / 1024).toFixed(2)}MB). Maximum size is 10MB.`);
        skippedCount++;
        continue;
      }

      validImageCount++;

      const promise = new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
          const result = event.target?.result;
          if (typeof result === 'string') {
            console.log(`Successfully read image ${file.name}, data URL length: ${result.length}`);
            resolve(result);
          } else {
            reject(new Error('Failed to read image'));
          }
        };
        reader.onerror = (error) => {
          console.error(`Error reading ${file.name}:`, error);
          reject(new Error(`Failed to read ${file.name}`));
        };
        reader.readAsDataURL(file);
      });
      
      imagePromises.push(promise);
    }

    if (imagePromises.length === 0) {
      console.warn(`No valid images found. Valid: ${validImageCount}, Skipped: ${skippedCount}`);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      return;
    }

    console.log(`Processing ${imagePromises.length} image(s)...`);
    Promise.all(imagePromises)
      .then((newImages) => {
        console.log(`Successfully loaded ${newImages.length} image(s)`);
        setImages((prev) => {
          const updated = [...prev, ...newImages];
          // Limit to 5 images total
          if (updated.length > 5) {
            console.warn(`Limiting to 5 images. Had ${updated.length}, keeping first 5.`);
            toast.warning(`Maximum 5 images allowed. Only the first 5 will be used.`);
            return updated.slice(0, 5);
          }
          console.log(`Total images now: ${updated.length}`);
          toast.success(`${newImages.length} image${newImages.length > 1 ? 's' : ''} added`);
          return updated;
        });
      })
      .catch((error) => {
        console.error('Error reading images:', error);
        toast.error('Failed to load images. Please try again.');
      });

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeImage = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    const hasMessage = message.trim().length > 0;
    const hasImages = images.length > 0;
    
    console.log('Submit clicked:', { hasMessage, hasImages, imagesCount: images.length, disabled });
    
    if ((hasMessage || hasImages) && !disabled) {
      console.log('Sending message with images:', { message: message.trim(), imagesCount: images.length });
      onSend(message.trim(), hasImages ? images : undefined);
      setMessage('');
      setImages([]);
    } else {
      console.log('Submit blocked:', { hasMessage, hasImages, disabled });
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled && images.length < 5) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled || images.length >= 5) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    // Create a synthetic event for handleImageSelect
    const syntheticEvent = {
      target: {
        files: files,
      },
    } as ChangeEvent<HTMLInputElement>;

    handleImageSelect(syntheticEvent);
  };

  return (
    <div 
      className="border-t border-border bg-background"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="max-w-3xl mx-auto p-4">
        {isDragging && (
          <div className="mb-3 p-4 border-2 border-dashed border-primary rounded-lg bg-primary/10 text-center">
            <p className="text-primary font-medium">Drop images here to upload</p>
          </div>
        )}
        {/* Preview selected images */}
        {images.length > 0 && (
          <div className="mb-3">
            <div className="flex flex-wrap gap-2">
              {images.map((image, index) => (
                <div key={index} className="relative group">
                  <img
                    src={image}
                    alt={`Preview ${index + 1}`}
                    className="w-24 h-24 object-cover rounded-lg border-2 border-border hover:border-primary/50 transition-colors"
                  />
                  <button
                    onClick={() => removeImage(index)}
                    className="absolute -top-2 -right-2 bg-destructive text-destructive-foreground rounded-full p-1.5 shadow-lg hover:bg-destructive/90 transition-colors"
                    aria-label="Remove image"
                    title="Remove image"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
            {images.length >= 5 && (
              <p className="text-xs text-muted-foreground mt-2">
                Maximum 5 images allowed
              </p>
            )}
          </div>
        )}
        
        <div className="relative flex items-end gap-3 bg-input rounded-xl border border-border focus-within:border-primary/50 transition-colors">
          <label
            htmlFor="chat-image-upload"
            className={`p-3 text-muted-foreground hover:text-foreground hover:bg-accent rounded-lg disabled:opacity-30 disabled:cursor-not-allowed transition-colors relative flex-shrink-0 cursor-pointer ${
              disabled || images.length >= 5 ? 'opacity-30 cursor-not-allowed' : ''
            }`}
            title={images.length >= 5 ? "Maximum 5 images allowed" : "Click to upload images (PNG, JPEG, GIF, WebP, BMP, SVG)"}
          >
            <ImageIcon className="w-5 h-5" />
            {images.length > 0 && (
              <span className="absolute -top-1 -right-1 bg-primary text-primary-foreground text-xs rounded-full w-5 h-5 flex items-center justify-center font-medium">
                {images.length}
              </span>
            )}
          </label>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={(e) => {
              console.log('File input onChange triggered', e);
              console.log('Files:', e.target.files);
              handleImageSelect(e);
            }}
            className="sr-only"
            disabled={disabled || images.length >= 5}
            id="chat-image-upload"
          />
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={images.length > 0 ? "Ask about the images or type a message..." : "Message RAG Assistant... (You can also upload images)"}
            disabled={disabled}
            rows={1}
            className="flex-1 bg-transparent text-foreground placeholder:text-muted-foreground resize-none py-3 px-4 focus:outline-none scrollbar-thin disabled:opacity-50"
            style={{ minHeight: '48px', maxHeight: '200px' }}
          />
          <button
            onClick={handleSubmit}
            disabled={(!message.trim() && images.length === 0) || disabled}
            className="p-3 text-muted-foreground hover:text-foreground disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        <p className="text-xs text-center text-muted-foreground mt-2">
          RAG Assistant can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
};

export default ChatInput;
